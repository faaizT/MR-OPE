import torch
import logging
logging.basicConfig(level = logging.INFO)
from tqdm import tqdm
import torch.optim as optim


def fit_weights_for_ate(
    model,
    data_loader,
    max_iters=250,
    lr=0.001,
    wd=0.0
):
    """Fitting weights for ATE computation, to the data we have in the behavioural dataset
    Args:
        model (Torch.nn): NN that outputs logits of the actions
        data_loader (torch.data.DataLoader): Dataloader that contains the (normalised) datasets (X, A, Y, policy_ratio)
        max_iters (int) maximum iterations to train the policy and reward
        lr (float) Learning rate
        wd (float) weight decay
    Return:
        TRAINED model, and model_reward
    """

    running_loss_log_vec = []
    # Find optimal model hyperparameters
    model.train()
    # Use the adam optimizer
    # NOTE: Changed optimizer to SGD from Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    loss_func = torch.nn.MSELoss()
    for epoch in range(max_iters):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            x, a, y, policy_ratio = data
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            x_and_y_concat = torch.cat([x, y], -1)
            output = model(x_and_y_concat)
            # Calc loss and backprop gradients
            
            # TODO CHECK THIS PART
            loss = loss_func(output, policy_ratio.detach())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            logging.info("[%d, %5d] MSEloss: %.3f" % (epoch + 1, i + 1, running_loss / len(data_loader)))
            running_loss_log_vec.append(running_loss / len(data_loader))
    model.eval()
    return model, running_loss_log_vec


def fit_weights(
    model,
    behav_policy,
    target_policy,
    data_loader,
    max_iters=250,
    lr=0.001,
):
    """Fitting weights to the data we have in the behavioural dataset
    Args:
        model (Torch.nn): NN that outputs logits of the actions
        behav_policy (Torch.nn): NN that represents behaviour policy
        target_policy (Torch.nn): NN that represents target policy
        data_loader (torch.data.DataLoader): Dataloader that contains the datasets (X, A, Y)
        max_iters (int) maximum iterations to train the policy and reward
        lr (float) Learning rate
        wd (float) weight decay
    Return:
        TRAINED model, and model_reward
    """

    running_loss_log_vec = []
    # Find optimal model hyperparameters
    model.train()
    # Use the adam optimizer
    # NOTE: Changed optimizer to SGD from Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.0
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    loss_func = torch.nn.MSELoss()
    for epoch in range(max_iters):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            x, a, y = data
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            policy_ratio = (
                target_policy(x)[torch.arange(x.shape[0]), a.reshape(-1)]
                / behav_policy(x)[torch.arange(x.shape[0]), a.reshape(-1)]
            ).unsqueeze(1)
            x_and_y_concat = torch.cat([x, y], -1)
            output = model(x_and_y_concat)
            # Calc loss and backprop gradients
            loss = loss_func(output, policy_ratio)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 100 == 0:
            logging.info("[%d, %5d] MSEloss: %.3f" % (epoch + 1, i + 1, running_loss / len(data_loader)))
            running_loss_log_vec.append(running_loss / len(data_loader))
    model.eval()
    return model, running_loss_log_vec





class MC_weights():
    def __init__(self, behav_policy, target_policy, batch_size, num_actions, Data_generator, mc_actions=5000, cts=False):
        self.behav_policy = behav_policy
        self.target_policy = target_policy
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.Data_generator = Data_generator
        self.mc_actions = mc_actions
        self.cts = cts

    def __call__(self, x, y):
        py_x_tar = 0
        py_x_beh = 0
        for a in range(self.num_actions):
            py_x_tar += self.target_policy(x)[:, a]*self.Data_generator._pY_ax(a, x)(y).reshape(-1)
            py_x_beh += self.behav_policy(x)[:, a]*self.Data_generator._pY_ax(a, x)(y).reshape(-1)
        return py_x_tar/py_x_beh

    def mc_estimator(self, x, y):
        # x is N x 1
        batches = int(self.mc_actions / self.batch_size)
        x = x.unsqueeze(1).repeat(1, self.batch_size, 1)
        x = x.reshape(-1, 1)
        y = y.unsqueeze(1).repeat(1, self.batch_size, 1)
        y = y.reshape(-1, 1)

        behav_pdf = 0
        target_pdf = 0
        for batch in tqdm(range(batches)):
            if self.cts:
                a_sampled_behav = self.behav_policy.sample(x).reshape(-1, 1)
                a_sampled_target = self.target_policy.sample(x).reshape(-1, 1)
            else:
                a_sampled_behav = torch.argmax(self.behav_policy.sample(x), -1).long().reshape(-1, 1)
                a_sampled_target = torch.argmax(self.target_policy.sample(x), -1).long().reshape(-1, 1)

            # sum over the actipon dimension
            behav_pdf += self.Data_generator._pY_ax(a_sampled_behav, x)(y).reshape(-1, self.batch_size).sum(1)
            target_pdf += self.Data_generator._pY_ax(a_sampled_target, x)(y).reshape(-1, self.batch_size).sum(1)
        return target_pdf/behav_pdf


def fit_behav_policy(
    model,
    data_loader,
    max_iters=100,
    lr=0.001,
    cts=False
):
    """Fitting a policy to the data we have in the behavioural dataset

    Args:
        model (Torch.nn): NN that outputs logits of the actions
        data_loader (torch.data.DataLoader): Dataloader that contains the datasets (X, A, Y)
        max_iters (int) maximum iterations to train the policy
    Return:
        TRAINED model, and model_reward
    """

    ################################################################
    # Training the behavioural policy network
    ################################################################
    if cts:
        criterion = torch.nn.GaussianNLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    len_data_loader = len(data_loader)
    for epoch in range(max_iters):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            x, a, y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if cts:
                outputs = model(x)
                loss = criterion(outputs[:, 0].reshape(-1, 1), a, torch.exp(outputs[:, 1].reshape(-1, 1)))
            else:
                outputs = model(x)
                loss = criterion(outputs.log(), a.reshape(-1))
            loss.backward()
            optimizer.step()

            # logging.info statistics
            running_loss += loss.item()
        if epoch % 20 == 0:
            logging.info(
                "[%d, %5d] POLICY loss: %.3f"
                % (epoch + 1, i + 1, running_loss / len(data_loader))
            )
    return model


def fit_ey_x_model(model, data_loader, max_iters, lr, combine_actions=False):
    """
    Fit a model for E[Y(a)|X]
    """    
    running_loss_log_vec = []
    # Find optimal model hyperparameters
    model.train()
    # Use the adam optimizer
    # NOTE: Changed optimizer to SGD from Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.0
    )  # Includes GaussianLikelihood parameters

    loss_func = torch.nn.MSELoss()
    for epoch in range(max_iters):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            x, a, y = data
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            if combine_actions:
                output = model(torch.cat([x, a], -1))
            else:
                output = model(x)
            # Calc loss and backprop gradients
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            logging.info("[%d, %5d] MSEloss for E[Y(a)|X]: %.3f" % (epoch + 1, i + 1, running_loss / len(data_loader)))
            running_loss_log_vec.append(running_loss / len(data_loader))
    model.eval()
    return model, running_loss_log_vec
