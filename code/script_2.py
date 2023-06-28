# Import libraries
# chaospy: generates pseudorandom sequences (e.g., Hammersley sequence)
import chaospy

# torch: a GPU-enabled vector library for machine learning and optimization tasks
import torch
import torch.optim as optim

# matplotlib: libraries for data visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# pandas and numpy: libraries for data manipulation and numerical computations
import pandas as pd
import numpy as np

# Set the data type and device based on availability of CUDA for torch
dtype = torch.double
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Pandas options: Disable chained assignment warning
pd.options.mode.chained_assignment = None

# Set the random seed for numpy and torch to ensure reproducibility
np.random.seed(123456789)
torch.manual_seed(123456789)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the output folder path
output_folder = './output/params/notef/'

# Data preprocessing
# Load the dataset
data = pd.read_csv('data/dpsdata.csv')

# Create additional columns 'zeros' and 'ones'
data["zeros"] = np.zeros_like(data.p.values)
data["ones"] = np.ones_like(data.p.values)
data = data[(data['urb'] == 'Rural') | (data['urb'] == 'Urban')]

# Create dataframes for different transactions
ca_data = data[['e0', 'c0', 'r0', 'zeros', 'ec0']]
dc_data = data[['e1', 'c1', 'r1', 'zeros', 'ec1']]
cc_data = data[['e2', 'c2', 'r2', 'rew', 'ec2']]

# Convert the dataframes to tensors
X_ca = torch.tensor(ca_data.values, dtype=dtype, device=device)
X_dc = torch.tensor(dc_data.values, dtype=dtype, device=device)
X_cc = torch.tensor(cc_data.values, dtype=dtype, device=device)

# Prepare demographic variables
# Convert categorical variables to dummy variables and normalize 'incl'
dem_data = data[['empd', 'mard', 'age', 'incl', 'gen',  'edu', 'ttype', 'urb', 'own', 'can', 'year', 'sph']]
dem_data_t = pd.get_dummies(dem_data, columns=['edu', 'ttype', 'year', 'urb']) # Convert categorical to dummies and drop the base cases
dem_data_t = dem_data_t.drop(['edu_Some/completed public school/some high school', 'ttype_Groceries/Drugs', 'year_2009', 'urb_Rural'], axis=1)
dem_data_t['const'] = data['ones']
dem_data_t['incl'] = dem_data_t['incl'] / 10000

# Convert demographic data to a tensor
Zbj = torch.tensor(dem_data_t.values, dtype=dtype, device=device)

# Create tensor for CSb
cs = torch.tensor(data.groupby('nid').mean()['cr_sc_avg'].values / 100, dtype=dtype, device=device).unsqueeze(1)

# Create tensors for CSb
CSb = torch.cat((cs, cs), 1)

# Merchant acceptance probabilities (transaction level)
Ms_0 = torch.tensor(data[['Ms0']].values, dtype=dtype, device=device)
Ms_1 = torch.tensor(data[['Ms01']].values, dtype=dtype, device=device)
Ms_2 = torch.tensor(data[['Ms012']].values, dtype=dtype, device=device)

# Consumer usage dummies for payment method (transaction level)
di = torch.tensor(data[['d0', 'd1', 'd2']].values, dtype=dtype, device=device)

# Convert Mb values to numeric representations
data['Mb'] = data['Mb'].map({
    'Credit card': 0,
    'Debit': 1,
    'Cash only': 2
})

# Consumer adoption dummies for payment bundles (consumer level)
BDi = torch.tensor(pd.get_dummies(data.groupby('nid').mean()['Mb']).values, dtype=dtype, device=device)

# Person ID (used to aggregate transaction level utility to individual level utility)
nid = torch.tensor(data['nid'].values, dtype=dtype, device=device)
nid_index = torch.nonzero(nid[..., None] == nid.unique())[:,1]
nid_matrix = torch.zeros(nid.unique().shape[0], nid.shape[0], dtype=dtype, device=device)
nid_matrix[nid_index, torch.arange(nid_index.shape[0])] = 1

# Separate data into groups based on adopted bundles by buyers
Mb = torch.tensor(data['Mb'].values, dtype=dtype, device=device)
group_0_idx = (Mb == 0)
group_1_idx = (Mb == 1)
group_2_idx = (Mb == 2)

# Suppresses random coefficients
beta_mask = torch.tensor([0, 1, 1, 0, 1], dtype=dtype, device=device)

# Placeholder for merchant acceptance with all 0
zero = torch.tensor([0.], dtype=dtype, device=device)

# Extract 'year' column
year = torch.tensor(data['year'], dtype=dtype, device=device)


# Function to calculate the choice of payment method in the second stage.
def stage2_choice(delta_ca, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2):
    # This tensor gathers all delta values for the three payment methods
    utilities = torch.stack((delta_ca, delta_dc, delta_cc), dim=1)

    # This tensor gathers all acceptance probabilities for the three payment methods
    probabilities = torch.stack((Ms_0, Ms_1, Ms_2), dim=1)

    # Multiply the utilities and probabilities to obtain a new tensor
    tensor_prod = utilities * probabilities

    # Identify the maximum values along the tensor for each individual
    max_values = tensor_prod.max(dim=1)[0]

    # Find the indices of the maximum values (i.e., the chosen payment methods)
    choices = torch.where(tensor_prod == max_values.unsqueeze(1), 1, 0)
    return choices


# Function to compute the expected maximum utility from the second stage.
def stage2_bundle_emax(delta_ca, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2):
    # Calculate the softmax (exponential of each element divided by the sum of the exponentials of all elements)
    # It's a way to convert raw scores into probabilities
    emax = torch.logsumexp(torch.stack((delta_ca + torch.log(Ms_0),
                                        delta_dc + torch.log(Ms_1),
                                        delta_cc + torch.log(Ms_2)), dim=1), dim=1)
    return emax


# Function to calculate the average usage probability and expected utility in the second stage.
def compute_avg_prob_util(mu, sd, alpha_ca, alpha_dc, alpha_cc, accept_0, accept_1, accept_2, draws):
    # Replicate `mu` along a new dimension equal to the size of `draws`
    mu_rep = mu.unsqueeze(1).repeat(1, len(draws))

    # Calculate the delta values for each payment method
    # Delta = utility function value for the individual, here represented as the sum of a mean utility (alpha)
    # and a normally distributed error term (mu + sd*draw)
    delta_ca = alpha_ca + mu_rep + sd * draws
    delta_dc = alpha_dc + mu_rep + sd * draws
    delta_cc = alpha_cc + mu_rep + sd * draws

    # Calculate the probability of acceptance for each payment method (softmax function applied)
    Ms_0 = torch.exp(delta_ca) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))
    Ms_1 = torch.exp(delta_dc) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))
    Ms_2 = torch.exp(delta_cc) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))

    # Calculate the expected maximum utility for the bundle of methods
    emax = stage2_bundle_emax(delta_ca, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2)

    # Return the average expected maximum utility (mean across the draws' dimension)
    return emax.mean(dim=1)


# Function to calculate the average usage probability and expected utility in the second stage without taking the exponent of `expsd`.
def compute_avg_prob_util_noexp(mu, expsd, alpha_ca, alpha_dc, alpha_cc, accept_0, accept_1, accept_2, draws):
    # Replicate `mu` along a new dimension equal to the size of `draws`
    mu_rep = mu.unsqueeze(1).repeat(1, len(draws))

    # Calculate the delta values for each payment method
    # Delta = utility function value for the individual, here represented as the sum of a mean utility (alpha)
    # and a normally distributed error term (mu + expsd*draw)
    delta_ca = alpha_ca + mu_rep + expsd * draws
    delta_dc = alpha_dc + mu_rep + expsd * draws
    delta_cc = alpha_cc + mu_rep + expsd * draws

    # Calculate the probability of acceptance for each payment method (softmax function applied)
    Ms_0 = torch.exp(delta_ca) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))
    Ms_1 = torch.exp(delta_dc) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))
    Ms_2 = torch.exp(delta_cc) / (torch.exp(delta_ca) + torch.exp(delta_dc) + torch.exp(delta_cc))

    # Calculate the expected maximum utility for the bundle of methods
    emax = stage2_bundle_emax(delta_ca, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2)

    # Return the average expected maximum utility (mean across the draws' dimension)
    return emax.mean(dim=1)

# This function calculates the standard error of the objective function with respect to the input using finite differences
def se_calc_finitediff(fobjective, inp, eps=1e-6):
    # Create an empty list to store the gradients
    grads = []

    # Iterate through each element in the input
    for i in range(len(inp)):
        # Create a copy of the input and decrease the ith element by eps
        inpminus = inp.clone()
        inpminus[i] = inpminus[i] - eps
        # Calculate the objective function with the decreased input
        objminus = fobjective(inpminus)

        # Create another copy of the input and increase the ith element by eps
        inpplus = inp.clone()
        inpplus[i] = inpplus[i] + eps
        # Calculate the objective function with the increased input
        objplus = fobjective(inpplus)

        # Compute the gradient approximation and append to the list of gradients
        grads.append((objplus - objminus) / (2 * eps))

    # Stack all the gradients together into a single tensor
    j = torch.stack(grads, 1)
    
    # Return the square root of the diagonal of the inverse of the dot product of the transposed gradient matrix with the gradient matrix
    # This is a way to estimate the standard error of the estimate of the objective function
    return j.t().mm(j).inverse().diag().sqrt()


# This function returns the log-likelihood of the second stage given the predicted choice probability and the observed outcome
def stage2_loglik(choice_prob, observed_outcome):
    # Negative log of the sum of observed outcomes powered by the choice probability
    # This is essentially the negative log likelihood of the observed outcomes given the choice probabilities
    return -torch.log(torch.pow(choice_prob, observed_outcome)).sum(1)


# This function returns the log-likelihood of the first stage given the predicted and observed data
def stage1_loglik(F, gamma, util, BDi):
    # Concatenate zeros to the tensor F along the column (dim=0)
    fixed_cost = torch.cat((zero, F))
    # Concatenate zeros to the CSb * gamma along the column (dim=1)
    variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma), 1)
    # Add util, fixed_cost and variable_cost tensors
    util_after_cost = util + fixed_cost + variable_cost
    # Compute the softmax of the util_after_cost tensor along columns. This gives the predicted probabilities for each choice
    likelihood = torch.nn.functional.softmax(util_after_cost, dim=1)
    # Negative log of the sum of observed outcomes powered by the predicted probabilities
    # This is essentially the negative log likelihood of the observed outcomes given the predicted probabilities
    return -torch.log(torch.pow(likelihood, BDi)).sum(1)


# Function to load parameters from a file
def get_params_from(filename):
    # Load data from a .npy file, which is a binary format for storing numpy arrays on disk
    p = np.load(output_folder + filename).item()
    # Convert the arrays into PyTorch tensors, which can be used to compute gradients
    mu = torch.tensor(p['beta'], dtype=dtype, device=device, requires_grad=True)
    # Check if the 'sd' key exists in the dictionary, if not, create a zeros tensor
    if 'sd' in p.keys():
        sd = torch.tensor(p['sd'], dtype=dtype, device=device, requires_grad=True)
    else:
        sd = torch.zeros(X_ca.shape[1], dtype=dtype, device=device, requires_grad=True)
    # Convert the rest of the arrays into PyTorch tensors
    alpha_ca = torch.tensor(p['alpha_ca'], dtype=dtype, device=device, requires_grad=True)
    alpha_dc = torch.tensor(p['alpha_dc'], dtype=dtype, device=device, requires_grad=True)
    alpha_cc = torch.tensor(p['alpha_cc'], dtype=dtype, device=device, requires_grad=True)
    F = torch.tensor(p['F'], dtype=dtype, device=device, requires_grad=True)
    if 'gamma' in p.keys():
        gamma = torch.tensor(p['gamma'], dtype=dtype, device=device, requires_grad=True)
    else:
        gamma = torch.zeros(CSb.shape[1], 1, dtype=dtype, device=device, requires_grad=True)
    draws = torch.tensor(p['draws'], dtype=dtype, device=device, requires_grad=True)
    ll = torch.tensor(p['ll'], dtype=dtype, device=device, requires_grad=True)
    return mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, ll


# Function to save the learned parameters, except the standard error, to a file
def save_param_nose_to(filename, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l):
    # Save the learned parameters (detached from the computation graph and moved to cpu) as numpy arrays in a dictionary to a .npy file
    np.save(output_folder + filename, {
      'beta': mu.detach().cpu().numpy(),
      'sd': sd.detach().cpu().numpy(),
      'alpha_ca': alpha_ca.detach().cpu().numpy(),
      'alpha_dc': alpha_dc.detach().cpu().numpy(),
      'alpha_cc': alpha_cc.detach().cpu().numpy(),
      'F': F.detach().cpu().numpy(),
      'gamma': gamma.detach().cpu().numpy(),
      'draws': draws.detach().cpu().numpy(),
      'll': l.detach().cpu().numpy(),
    })



# Function to save all learned parameters, including the standard error, to a file
def save_param_to(filename, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, mu_se, sd_se, alpha_dc_se, alpha_cc_se, F_se, gamma_se, draws, ll):
    # Save the learned parameters and their standard errors (detached from the computation graph and moved to cpu) as numpy arrays in a dictionary to a .npy file
    np.save(output_folder + filename, {
      'beta': mu.detach().cpu().numpy(),
      'sd': sd.detach().cpu().numpy(),
      'alpha_ca': alpha_ca.detach().cpu().numpy(),
      'alpha_dc': alpha_dc.detach().cpu().numpy(),
      'alpha_cc': alpha_cc.detach().cpu().numpy(),
      'F': F.detach().cpu().numpy(),
      'gamma': gamma.detach().cpu().numpy(),
      'beta_se': mu_se.detach().cpu().numpy(),
      'sd_se': sd_se.detach().cpu().numpy(),
      'alpha_dc_se': alpha_dc_se.detach().cpu().numpy(),
      'alpha_cc_se': alpha_cc_se.detach().cpu().numpy(),
      'F_se': F_se.detach().cpu().numpy(),
      'gamma_se': gamma_se.detach().cpu().numpy(),
      'draws': draws.detach().cpu().numpy(),
      'll': ll.detach().cpu().numpy(),
    })



# The following functions are for calculating the log likelihood of each parameter individually. This is necessary for calculating the standard error.

# Function for log likelihood calculation with respect to mu
def ll2mu(mu1):
    # Call the 'fit' function with 'mu1' and the rest of the parameters fixed
    return fit(mu1, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws)

# Function for log likelihood calculation with respect to sd
def ll2sd(sd1):
    # Call the 'fit' function with 'sd1' and the rest of the parameters fixed
    return fit(mu, sd1, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws)

# Function for log likelihood calculation with respect to alpha_dc
def ll2dc(alpha_dc1):
    # Call the 'fit' function with 'alpha_dc1' and the rest of the parameters fixed
    return fit(mu, sd, alpha_ca, alpha_dc1, alpha_cc, F, gamma, draws)

# Function for log likelihood calculation with respect to alpha_cc
def ll2cc(alpha_cc1):
    # Call the 'fit' function with 'alpha_cc1' and the rest of the parameters fixed
    return fit(mu, sd, alpha_ca, alpha_dc, alpha_cc1, F, gamma, draws)

# Function for log likelihood calculation with respect to F
def ll1F(F1):
    # Call the 'fit' function with 'F1' and the rest of the parameters fixed
    return fit(mu, sd, alpha_ca, alpha_dc, alpha_cc, F1, gamma, draws)

# Function for log likelihood calculation with respect to gamma
def ll1betaF(gamma1):
    # Call the 'fit' function with 'gamma1' and the rest of the parameters fixed
    return fit(mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma1, draws)


# This function is responsible for training the model. 
# It uses an optimization algorithm and a learning rate scheduler to iteratively adjust the model parameters to best fit the training data.
def fit(optimizer, scheduler, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws):
    ls = []  # List to hold the loss value at each iteration
    
    # Perform the specified number of iterations
    for i in range(opt['iter']):
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        
        # Compute average choice probabilities and utility given current parameter estimates
        avg_choice, avg_util = compute_avg_prob_util(mu, sd, alpha_ca, alpha_dc, alpha_cc, Ms_0, Ms_1, Ms_2, draws)
        
        # Calculate the log-likelihood, i.e., the objective function we're trying to maximize
        l = torch.sum(stage2_loglik(avg_choice, di)) + torch.sum(stage1_loglik(F, gamma, avg_util, BDi))
        
        ls.append(float(l))  # Append the current log-likelihood to the list
        
        # Compute the gradients of the log-likelihood with respect to the parameters
        l.backward()
        
        # Perform a step of the optimization algorithm
        optimizer.step()
        
        # Step the scheduler, this can adjust the learning rate based on the current state of training
        scheduler.step(l)
        
        # Print progress every 500 iterations
        if (i+1) % (500) == 0:
            print('------------------------------------------------------')
            print("LL", l.detach().cpu().numpy().round(4))
            print("Mu", mu.detach().cpu().numpy().round(4))
            print("SD", torch.exp(sd).detach().cpu().numpy().round(4))
            print("F", F.detach().cpu().numpy().round(4))
            print("beta F", gamma.detach().cpu().numpy().round(4))
            print('------------------------------------------------------')
            print()

        # If the learning rate falls below a certain threshold, stop training and declare convergence
        if scheduler.optimizer.param_groups[0]['lr'] < 1e-7:
            print('Converged')
            break
            
        # If the log-likelihood becomes NaN (not a number), stop training
        if torch.isnan(l):
            print('NAN')
            break
    
    # Print final joint log-likelihood
    print("Joint LL: {:.3f}".format(l))
    
    # Plot the log-likelihood at each iteration
    plt.plot(ls)
    plt.show()
    
    # Return final log-likelihood
    return l

# Specification 1 - Conditional Logit

# This code block represents Specification 1 of the script, which uses the Conditional Logit model.
# The opt dictionary specifies the number of iterations, learning rate, and other optimization parameters.
# The draws tensor is initialized with zeros, representing the random draws used in the model.
# Model parameters are defined using tensors, such as mu, sd, alpha_ca, alpha_dc, alpha_cc, F, and gamma. They are randomly initialized with appropriate shapes and device assignments.
# An optimizer (Adam) and scheduler (ReduceLROnPlateau) are created to optimize the model parameters during training.
# The fit function is called to train the model, passing the optimizer, scheduler, model parameters, and draws tensor.
# After training, the learned parameters are saved to a file using `save_param_nose_to`.


opt = {'iter': 10000, 'lr': 0.001, 'amsgrad': True}

# Initialize draws tensor
draws = torch.zeros(1, nid.unique().shape[0], X_ca.shape[1], dtype=dtype, device=device)[:, nid_index, :]

# Define model parameters
mu = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)
sd = torch.zeros(X_ca.shape[1], dtype=dtype, requires_grad=False, device=device)
alpha_ca = torch.zeros(Zbj.shape[1], dtype=dtype, requires_grad=False, device=device)
alpha_dc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
alpha_cc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
F = torch.randn(CSb.shape[1], dtype=dtype, requires_grad=True, device=device)
gamma = torch.zeros(CSb.shape[1], dtype=dtype, requires_grad=False, device=device)

# Create optimizer and scheduler
optimizer = optim.Adam([mu, alpha_dc, alpha_cc, F], opt['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=1e-8, threshold_mode='abs')

# Fit the model
l = fit(optimizer, scheduler, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws)

# Save parameters (excluding standard error)
save_param_nose_to('spec1_asc', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l)

# Load parameters and calculate standard errors
mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l = get_params_from('spec1_asc.npy')
mu_se = se_calc_finitediff(ll2mu, mu, eps=1e-12)
sd_se = torch.tensor([0, 0, 0, 0, 0])
alpha_dc_se = se_calc_finitediff(ll2dc, alpha_dc, eps=1e-12)
alpha_cc_se = se_calc_finitediff(ll2cc, alpha_cc, eps=1e-12)
F_se = se_calc_finitediff(ll1F, F, eps=1e-12)
gamma_se = se_calc_finitediff(ll1betaF, gamma, eps=1e-12)

# Save parameters (including standard error)
save_param_to('spec1_asc', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, mu_se, sd_se, alpha_dc_se, alpha_cc_se, F_se, gamma_se, draws, l)


# Specification 2 - Mixed Logit
opt = {'iter': 100000, 'lr': 0.01, 'num_sample':100, 'amsgrad': True}  # define the optimization options

# using chaospy to generate normal distribution samples
base_dist = chaospy.Normal(0,1)  # define a normal distribution with mean=0 and std=1

# create an identical and independent (iid) distribution with dimensions equal to the column size of X_ca
dist = chaospy.Iid(base_dist, X_ca.shape[1])  

# generate Hammersley sequence samples with dimensions and order (number of samples) defined, reshape them to match our required shape
samples = chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples(
    order=opt['num_sample']*nid.unique().shape[0], 
    dim=X_ca.shape[1], 
    burnin=-1
)

# convert the samples from Hammersley sequence (which are uniform in nature) to follow our defined normal distribution
draws = torch.tensor(
    dist.inv(samples).T.reshape(opt['num_sample'], nid.unique().shape[0], X_ca.shape[1]), 
    dtype=dtype, 
    device=device
)[:, nid_index, :]  # select only samples that corresponds to nid_index

# Define initial parameters for the model. All are randomly initialized except for alpha_ca and gamma which are set to zero
mu = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)  # parameters for the conditional average (CA) model
sd = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)  # parameters for the standard deviation of the CA model
alpha_ca = torch.zeros(Zbj.shape[1], dtype=dtype, requires_grad=False, device=device)  # parameters for the constant average (CA) model
alpha_dc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)  # parameters for the deterministic choice (DC) model
alpha_cc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)  # parameters for the constant choice (CC) model
F = torch.randn(2, dtype=dtype, requires_grad=True, device=device)  # parameters for the fixed cost in the utility function
gamma = torch.zeros(CSb.shape[1], dtype=dtype, requires_grad=False, device=device)  # parameters for the variable cost in the utility function

# Create an ADAM optimizer with the parameters and the learning rate defined in opt dictionary
optimizer = optim.Adam([mu, sd, alpha_dc, alpha_cc, F], opt['lr'])

# Create a learning rate scheduler that reduces the learning rate when the loss does not improve
# The learning rate is reduced by a factor of 0.5 (factor parameter) if there's no improvement for 100 epochs (patience parameter)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=1e-8, threshold_mode='abs')

# Fit the model using the defined optimizer, scheduler and initial parameters
l = fit(optimizer, scheduler, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws)

# Save the trained parameters and the loss into a file
save_param_nose_to('spec2_mix', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l)

# Load the parameters from a file
mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l = get_params_from('spec2_mix.npy')

# Use finite difference to calculate the standard error (SE) of each parameter
# We use a very small epsilon (1e-12) for the finite difference calculation
mu_se = se_calc_finitediff(ll2mu, mu, eps=1e-12)
sd_se = se_calc_finitediff(ll2sd, torch.exp(sd[[1,2,4]]), eps=1e-12)
# We only need the SE for certain elements of sd, the rest are set to zero
sd_se = torch.tensor([0, sd_se[0], sd_se[1], 0, sd_se[2]])
alpha_dc_se = se_calc_finitediff(ll2dc, alpha_dc, eps=1e-12)
alpha_cc_se = se_calc_finitediff(ll2cc, alpha_cc, eps=1e-12)
F_se = se_calc_finitediff(ll1F, F, eps=1e-12)
gamma_se = se_calc_finitediff(ll1betaF, gamma, eps=1e-12)

# Save all the parameters, including the SEs, and the loss into a file
save_param_to('spec2_mix', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, mu_se, sd_se, alpha_dc_se, alpha_cc_se, F_se, gamma_se, draws, l)

# Specification 3 - Mixed Logit with Credit Score

# This code block represents Specification 3 of the script, which uses the Mixed Logit model with Credit Score.
# The opt dictionary specifies the number of iterations, learning rate, number of samples, and other optimization parameters.

opt = {'iter': 10000, 'lr': 0.1, 'num_sample':100, 'amsgrad': True}

# Draw Hammersley sequences
base_dist = chaospy.Normal(0,1)
dist = chaospy.Iid(base_dist, X_ca.shape[1])
samples = chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples(order=opt['num_sample']*nid.unique().shape[0], dim=X_ca.shape[1], burnin=-1)
draws = torch.tensor(dist.inv(samples).T.reshape(opt['num_sample'], nid.unique().shape[0], X_ca.shape[1]), dtype=dtype, device=device)[:, nid_index, :]

# Model parameters are defined using tensors, such as mu, sd, alpha_ca, alpha_dc, alpha_cc, F, and gamma.
# They are randomly initialized with appropriate shapes and device assignments.

mu = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)
sd = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)
alpha_ca = torch.zeros(Zbj.shape[1], dtype=dtype, requires_grad=False, device=device)
alpha_dc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
alpha_cc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
F = torch.randn(2, dtype=dtype, requires_grad=True, device=device)
gamma = torch.randn(CSb.shape[1], dtype=dtype, requires_grad=True, device=device)

# An optimizer (Adam) and scheduler (ReduceLROnPlateau) are created to optimize the model parameters during training.

optimizer = optim.Adam([mu, sd, alpha_dc, alpha_cc, F, gamma], opt['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=1e-8, threshold_mode='abs')

# The fit function is called to train the model, passing the optimizer, scheduler, model parameters, and draws tensor.

l = fit(optimizer, scheduler, mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws)

# After training, the learned parameters are saved to a file using `save_param_nose_to`.

save_param_nose_to('spec3_mix_cs', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l)

# The following code block loads the learned parameters and calculates their standard errors using `se_calc_finitediff`.

# Run this on CPU with a lot of memory

mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l = get_params_from('spec3_mix_cs.npy')
mu_se = se_calc_finitediff(ll2mu, mu, eps=1e-12)
sd_se = se_calc_finitediff(ll2sd, torch.exp(sd[[1,2,4]]), eps=1e-12)
sd_se = torch.tensor([0, sd_se[0], sd_se[1], 0, sd_se[2]])
alpha_dc_se = se_calc_finitediff(ll2dc, alpha_dc, eps=1e-12)
alpha_cc_se = se_calc_finitediff(ll2cc, alpha_cc, eps=1e-12)
F_se = se_calc_finitediff(ll1F, F, eps=1e-12)
gamma_se = se_calc_finitediff(ll1betaF, gamma, eps=1e-12)

# The learned parameters and their standard errors are saved to a file using `save_param_to`.

save_param_to('spec3_mix_cs', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, mu_se, sd_se, alpha_dc_se, alpha_cc_se, F_se, gamma_se, draws, l)

# Second Stage Only, Analyze util gain

# Take the counterfactual scenario of CBDC and computes re-normalized variables
def renormalize(X_bc):
    # Calculates the sum of all payment variables to use for renormalization
    X_sum = X_ca + X_dc + X_cc + X_bc

    # Computes re-normalized variables for each payment category
    X_ca_renorm = torch.cat(((X_ca / X_sum)[:,:3], X_ca[:,3:5]), 1)
    X_bc_renorm = torch.cat(((X_bc / X_sum)[:,:3], X_bc[:,3:5]), 1)
    X_dc_renorm = torch.cat(((X_dc / X_sum)[:,:3], X_dc[:,3:5]), 1)
    X_cc_renorm = torch.cat(((X_cc / X_sum)[:,:3], X_cc[:,3:5]), 1)

    return X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm

# The expected maximum utility of adopting a payment bundle
def stage1_emax_ctf(F, gamma, stage2_emax):
    # Computes the fixed cost and variable cost based on payment bundle attributes and credit score
    fixed_cost = torch.cat((zero, F))
    variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma), 1)

    # Computes the utility after considering the cost for each payment bundle
    util_after_cost = stage2_emax + fixed_cost + variable_cost

    # Calculates the expected maximum utility after considering the cost
    return torch.logsumexp(util_after_cost, dim=2, keepdim=True)

# The expected maximum utility of each transaction
def stage2_bundle_emax_ctf(delta_ca, delta_bc, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2):
    # Calculates the utility for each payment bundle separately
    u_0 = torch.logsumexp(torch.cat([delta_ca, delta_bc], 2), dim=2, keepdim=True)
    u_1 = torch.logsumexp(torch.cat([delta_ca, delta_bc, delta_dc], 2), dim=2, keepdim=True)
    u_2 = torch.logsumexp(torch.cat([delta_ca, delta_bc, delta_dc, delta_cc], 2), dim=2, keepdim=True)

    # Computes the expected maximum utility based on the acceptance probabilities of each payment bundle
    eu_0 = u_0
    eu_1 = accept_prob_0 * u_0 + (accept_prob_1 + accept_prob_2) * u_1
    eu_2 = accept_prob_0 * u_0 + accept_prob_1 * u_1 + accept_prob_2 * u_2

    # Calculates the utilities for each individual by aggregating transaction-level utilities
    u = torch.cat((eu_0, eu_1, eu_2), dim=2)
    utils = torch.matmul(u.transpose(1,2), nid_matrix.t()).transpose(1,2)

    return utils

# Computes adoption probability of each bundle
def adoption_prob_ctf(F, gamma, util):
    # Computes the fixed cost and variable cost based on payment bundle attributes and credit score
    fixed_cost = torch.cat((zero, F))
    variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma), 1)

    # Computes the utility after considering the cost for each payment bundle
    util_after_cost = util + fixed_cost + variable_cost

    # Calculates the likelihood of adopting each payment bundle based on the utilities
    likelihood = torch.nn.functional.softmax(util_after_cost, dim=2)

    return likelihood

# Computes usage probability of each payment method, taking into account adoption probability
def usage_prob_ctf(delta_ca, delta_bc, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2, adopt_prob_0, adopt_prob_1, adopt_prob_2):
    # Probabilities if bundle 0 is adopted
    p_0 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc], dim=2), dim=2)
    p_ca_0 = p_0[:,:,0:1]
    p_bc_0 = p_0[:,:,1:2]

    # Probabilities if bundle 1 is adopted
    p_1 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc, delta_dc], dim=2), dim=2)
    p_ca_1 = p_1[:,:,0:1]
    p_bc_1 = p_1[:,:,1:2]
    p_dc_1 = p_1[:,:,2:3]

    # Probabilities if bundle 2 is adopted
    p_2 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc, delta_dc, delta_cc], dim=2), dim=2)
    p_ca_2 = p_2[:,:,0:1]
    p_bc_2 = p_2[:,:,1:2]
    p_dc_2 = p_2[:,:,2:3]
    p_cc_2 = p_2[:,:,3:4]

    # Calculates the usage probability of each payment method
    p_ca = p_ca_0 * adopt_prob_0 + p_ca_0 * (adopt_prob_1 + adopt_prob_2) * accept_prob_0 \
            + p_ca_1 * adopt_prob_1 * (accept_prob_1 + accept_prob_2) \
            + p_ca_1 * adopt_prob_2 * accept_prob_1 \
            + p_ca_2 * adopt_prob_2 * accept_prob_2

    p_bc = p_bc_0 * adopt_prob_0 + p_bc_0 * (adopt_prob_1 + adopt_prob_2) * accept_prob_0 \
            + p_bc_1 * adopt_prob_1 * (accept_prob_1 + accept_prob_2) \
            + p_bc_1 * adopt_prob_2 * accept_prob_1 \
            + p_bc_2 * adopt_prob_2 * accept_prob_2

    p_dc = p_dc_1 * adopt_prob_1 * (accept_prob_1 + accept_prob_2) \
            + p_dc_1 * adopt_prob_2 * accept_prob_1 \
            + p_dc_2 * adopt_prob_2 * accept_prob_2

    p_cc = p_cc_2 * adopt_prob_2 * accept_prob_2

    return torch.cat((p_ca, p_bc, p_dc, p_cc), dim=2)

# Computes the probability of adoption and usage given counterfactual inputs
def compute_avg_prob_util_ctf(X0, X1, X2, Xbc, mu, sd,
                            alpha_ca, alpha_dc, alpha_cc, alpha_bc,
                            F, gamma, accept_0, accept_1, accept_2, draws):
    # Calculates the beta parameters based on draws, standard deviation, and mean parameters
    beta = draws * torch.exp(sd) * beta_mask + mu

    # Calculates the utility differences for each payment bundle
    delta_ca = (beta * X0).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
    delta_bc = (beta * Xbc).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
    delta_dc = (beta * X1).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
    delta_cc = (beta * X2).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)

    # Computes the average utility and adoption probability
    avg_util = stage2_bundle_emax_ctf(delta_ca, delta_bc, delta_dc, delta_cc, accept_0, accept_1, accept_2)
    adopt_prob = adoption_prob_ctf(F, gamma, avg_util)
    adopt_prob_expanded = adopt_prob[:, nid_index.long(), :]
    avg_choice = usage_prob_ctf(delta_ca, delta_bc, delta_dc, delta_cc,
                                          accept_0, accept_1, accept_2,
                                          adopt_prob_expanded[:, :, 0:1], adopt_prob_expanded[:, :, 1:2], adopt_prob_expanded[:, :, 2:3])

    return avg_choice.mean(0), adopt_prob.mean(0)

# best of both
# Set optimization parameters
opt = {'iter': 10000, 'lr': 0.1, 'num_sample': 100, 'amsgrad': True}

# Draw Hammersley sequences
base_dist = chaospy.Normal(0, 1)
dist = chaospy.Iid(base_dist, X_ca.shape[1])
samples = chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples(
    order=opt['num_sample'] * nid.unique().shape[0], dim=X_ca.shape[1], burnin=-1)
draws = torch.tensor(dist.inv(samples).T.reshape(opt['num_sample'], nid.unique().shape[0], X_ca.shape[1]),
                     dtype=dtype, device=device)[:, nid_index, :]

# Initialize model parameters
mu = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)
sd = torch.randn(X_ca.shape[1], dtype=dtype, requires_grad=True, device=device)
alpha_ca = torch.zeros(Zbj.shape[1], dtype=dtype, requires_grad=False, device=device)
alpha_dc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
alpha_cc = torch.randn(Zbj.shape[1], dtype=dtype, requires_grad=True, device=device)
F = torch.randn(2, dtype=dtype, requires_grad=False, device=device)
gamma = torch.randn(CSb.shape[1], dtype=dtype, requires_grad=False, device=device)

# Create optimizer and scheduler
optimizer = optim.Adam([mu, sd, alpha_dc, alpha_cc], opt['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, threshold=1e-8, threshold_mode='abs')

ls = []
for i in range(opt['iter']):
    optimizer.zero_grad()
    avg_choice, avg_util = compute_avg_prob_util(mu, sd,
                                                 alpha_ca, alpha_dc, alpha_cc,
                                                 Ms_0, Ms_1, Ms_2, draws)
    l = torch.sum(stage2_loglik(avg_choice, di))
    ls.append(float(l))
    l.backward()
    optimizer.step()
    scheduler.step(l)
    if (i + 1) % (500) == 0:
        print('------------------------------------------------------')
        print("LL", l.detach().cpu().numpy().round(4))
        print("Mu", mu.detach().cpu().numpy().round(4))
        print("SD", torch.exp(sd).detach().cpu().numpy().round(4))
        print("F", F.detach().cpu().numpy().round(4))
        print("beta F", gamma.detach().cpu().numpy().round(4))
        print('------------------------------------------------------')
        print()
    if scheduler.optimizer.param_groups[0]['lr'] < 1e-7:
        print('Converged')
        break
    if torch.isnan(l):
        print('NAN')
        break

print("Joint LL: {:.3f}".format(l))
plt.plot(ls)
plt.show()
save_param_nose_to('spec4_mix_cs_ss', mu, sd, alpha_ca, alpha_dc, alpha_cc, F, gamma, draws, l)

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:, 4] = X_dc[:, 4].clone()

# Re-normalize the perception variables
X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)
alpha_bc = alpha_dc

# Only in 2017
y = 2017
year_idx = (year == y)
# Compute the adoption utility WITHOUT CBDC
beta = draws[:, nid_index, :] * torch.exp(sd) * beta_mask + mu
delta_ca_nobc = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_dc_nobc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc_nobc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_nocbdc = stage2_bundle_emax(delta_ca_nobc, delta_dc_nobc, delta_cc_nobc, Ms_0, Ms_1, Ms_2)

# Compute the adoption utility WITH CBDC
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_cbdc = stage2_bundle_emax_ctf(delta_ca, delta_bc, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2)

# Convert into dollar and put it in a table
expect_util_improvement = ((stage2_emax_cbdc - stage2_emax_nocbdc).mean(0) * BDi).sum(1)[
    nid[year_idx].unique().long()] / float(-mu[4])
util_improvement_table = data[['nid', 'empd', 'mard', 'agec', 'incc', 'gen', 'edu',
                               'urb', 'own', 'can', 'year', 'sph']].drop_duplicates()
util_improvement_table = util_improvement_table[util_improvement_table.year == 2017]
util_improvement_table['u'] = expect_util_improvement.detach().cpu()

# Group by and take mean of util improvement by demographics
util_improv_edu = util_improvement_table.groupby('edu').mean()[['u']]
util_improv_edu['eduname'] = ['High school', 'Grad School', 'Some uni.', '< high school', 'Tech. school', 'Uni degree']
util_improv_edu = util_improv_edu.sort_values(by='u', ascending=False)

util_improv_age = util_improvement_table.groupby('agec').mean()[['u']]
util_improv_age['agen'] = ['18-32', '33-42', '43-52', '53-61', '62-99']
util_improv_age = util_improv_age.sort_values(by='u', ascending=False)

util_improv_inc = util_improvement_table.groupby('incc').mean()[['u']]
util_improv_inc['incn'] = ['< 25k', '25k-45k', '45k-65k', '65k-90k', '> 90k']
util_improv_inc = util_improv_inc.sort_values(by='u', ascending=False)

mean_imp = util_improvement_table['u'].mean()

util_improvement_table['u'].mean()

import matplotlib.gridspec as gridspec

# Plot the distribution and breakdown of util gain
gs = gridspec.GridSpec(2,3)
plt.figure(figsize=(16,12))

# Distribution of utility improvement
ax1 = plt.subplot(gs[0, :])
plt.hist(expect_util_improvement.detach().cpu(), bins=np.arange(float(expect_util_improvement.min().round()), float(expect_util_improvement.max().ceil()), 0.25), color='dimgrey')
plt.xlabel("Utility improvement in dollars")
plt.ylabel("Person count")

# Utility improvement by education
ax2 = plt.subplot(gs[1, 0])
plt.bar(x=util_improv_edu['eduname'], height=util_improv_edu['u'], align='center', color='dimgrey')
plt.ylabel("Utility improvement in dollars")
plt.xticks(rotation=90)

# Utility improvement by age
ax3 = plt.subplot(gs[1, 1], sharey=ax2)
plt.bar(x=util_improv_age['agen'], height=util_improv_age['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax3.get_yticklabels(), visible=False)

# Utility improvement by income
ax4 = plt.subplot(gs[1, 2], sharey=ax2)
plt.bar(x=util_improv_inc['incn'], height=util_improv_inc['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax4.get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()

# Save to a file
util_improvement_table.to_csv('ecs3cl.csv', index=False)

# Elasticity (on CPU)

X_ca_t = X_ca.clone()
X_dc_t = X_dc.clone()
X_cc_t = X_cc.clone()

year_idx = (year == 2017)
avg_choice, avg_util = compute_avg_prob_util(mu, sd,
                        alpha_ca, alpha_dc, alpha_cc,
                        Ms_0, Ms_1, Ms_2, draws)
fixed_cost = torch.cat((zero, F))
variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma), 1)
util_after_cost = avg_util + fixed_cost + variable_cost
likelihood = torch.nn.functional.softmax(util_after_cost, dim=1)

baseline_usage = avg_choice[year_idx]
baseline_adopt = likelihood[nid[year_idx].unique().long()]

eps = 1e-10
var_idxs = [0, 1, 2, 4]
var_name = ['Ease', 'Affordability', 'Risk', 'Tcost']
mb_changed = [X_ca, X_dc, X_cc]
mb_name = ['Cash', 'Debit', 'Credit']

# Compute elasticities for each variable and payment method combination
for j in range(len(var_idxs)):
    print("Changing variable", var_name[j])
    varidx = var_idxs[j]

    for i in range(len(mb_changed)):
        var_changed = mb_changed[i]
        print("  Changing", mb_name[i])
        # Compute the baseline value for elasticity
        baseline_var = var_changed[year_idx, varidx]
        baseline_var_pp = (torch.mm(nid_matrix, var_changed) / torch.mm(nid_matrix, torch.ones_like(Ms_0)))[nid[year_idx].unique().long(), varidx]

        # Change the value by eps and compute the probability of usage and adopt
        var_changed[:,varidx] = var_changed[:,varidx] + eps
        avg_choice, avg_util = compute_avg_prob_util(mu, sd,
                                alpha_ca, alpha_dc, alpha_cc,
                                Ms_0, Ms_1, Ms_2, draws)
        util_after_cost = avg_util + fixed_cost + variable_cost
        likelihood = torch.nn.functional.softmax(util_after_cost, dim=1)

        usage = avg_choice[year_idx]
        adopt = likelihood[nid[year_idx].unique().long()]

        # Compute the non-nan elasticity mean
        e_usage = []
        e_adopt = []
        for k in range(3):
            derivative = (usage - baseline_usage) / eps
            elasticities = derivative[:,k] * baseline_var / baseline_usage[:,k]

            # idx_notnan = 1 - torch.isnan(elasticities)
            idx_notnan = ~torch.isnan(elasticities)
            elasticities_nonan = elasticities[idx_notnan]
            e_usage.append(elasticities_nonan.mean(0).detach().cpu().numpy())

            derivative = (adopt - baseline_adopt) / eps
            elasticities = derivative[:,k] * baseline_var_pp / baseline_adopt[:,k]

            # idx_notnan = 1 - torch.isnan(elasticities)
            idx_notnan = ~torch.isnan(elasticities)
            elasticities_nonan = elasticities[idx_notnan]
            e_adopt.append(elasticities_nonan.mean(0).detach().cpu().numpy())
        print('    Elasticity of usage', np.array(e_usage).round(3))
        print('    Elasticity of adopt', np.array(e_adopt).round(3))
        var_changed[:,varidx] = var_changed[:,varidx] - eps

# Counterfactuals

# Graphical settings
plt.rcParams['figure.figsize'] = 8,6
plt.rcParams.update({'font.size': 18})

# Settings for counterfactuals

# Cash-like
# X_bc = X_ca.clone()

# Cheaper debit
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

# Re-normalize the perception variables
X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

# Only in 2017
y = 2017
year_idx = (year == y)

# Display observed market share pie chart
plt.pie(di[year_idx].mean(0).detach().cpu(),
        labels=['Cash', 'Debit', 'Credit'],
        colors=['#E69F00', '#009E73', '#CC79A7'],
        autopct='%3.1f')
plt.show()

# Compute counterfactual market share using learned parameters
mu, sd, alpha_ca, alpha_dc,\
    alpha_cc, F, gamma, draws, l = get_params_from('spec3_mix_cs.npy')

# Set CBDC mlogit params
alpha_bc = alpha_dc

avg_choice, avg_adopt = compute_avg_prob_util_ctf(X_ca_renorm, X_dc_renorm, X_cc_renorm, X_bc_renorm,
                        mu, sd, alpha_ca, alpha_dc, alpha_cc, alpha_bc, F, gamma,
                        Ms_0, Ms_1, Ms_2, draws)

print('Adopt prob', avg_adopt[nid[year_idx].unique().long()].mean(0).detach().cpu().numpy())
print('Usage prob', avg_choice[year_idx].mean(0).detach().cpu().numpy())

# Display counterfactual market share pie chart
plt.figure(figsize=(6,6))
plt.pie(avg_choice[year_idx].mean(0).detach().cpu(),
        labels=['Cash', 'CBDC', 'Debit', 'Credit'],
        colors=['#E69F00', '#56B4E9', '#009E73', '#CC79A7'],
        autopct='%3.1f')
plt.show()

### Contribution of each variable to utility
# Compute the utility from the X variables and population mean
util_ca = (X_ca * mu.repeat(Zbj.shape[0],1))[year_idx]
util_dc = (X_dc * mu.repeat(Zbj.shape[0],1))[year_idx]
util_cc = (X_cc * mu.repeat(Zbj.shape[0],1))[year_idx]
util_ca_mean = util_ca.mean(0)
util_dc_mean = util_dc.mean(0)
util_cc_mean = util_cc.mean(0)

print('Util mean CA', util_ca_mean.abs().detach().cpu().numpy())
print('Util mean DC', util_dc_mean.abs().detach().cpu().numpy())
print('Util mean CC', util_cc_mean.abs().detach().cpu().numpy())

util_share = torch.stack((util_ca_mean, util_dc_mean, util_cc_mean)).t()
idx = ['Cash', 'Debit', 'Credit']
label = ['Ease of use', 'Affordability of use', 'Security of use', 'Reward', 'Transaction cost']

dd = pd.DataFrame({label[0]: util_share[0,:].detach().cpu(),
                   label[1]: util_share[1,:].detach().cpu(),
                   label[2]: util_share[2,:].detach().cpu(),
                   label[3]: util_share[3,:].detach().cpu(),
                   label[4]: util_share[4,:].detach().cpu()}, index=idx)[label]
ax = dd.plot(kind='bar', stacked=True, width=0.35, color=['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#666666'])
ax.set_ylabel('Utility in dollars')
plt.hlines(0, -1, 3, linewidth=0.1)
plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), fancybox=True)
r = ax.set_xlim(-1., 3)
plt.show()

### How cheap/good does CBDC have to be vs cash?

# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

X_bc_base = X_bc.clone()
alpha_bc = alpha_dc

year_idx = (year == 2017)
start = -25
end = 30
step = 5

s_ca_cost, s_bc_cost, s_dc_cost, s_cc_cost = [], [], [], []
for multipler in range(start, end, step):
    X_bc = X_bc_base.clone()
    # Adjust the transaction cost based on the multiplier
    X_bc[:,4] = X_bc_base[:,4].clone() * (1 + multipler/100)
    X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)
    choice, adopt = compute_avg_prob_util_ctf(X_ca_renorm, X_dc_renorm, X_cc_renorm, X_bc_renorm, mu, sd,
                        alpha_ca, alpha_dc, alpha_cc, alpha_bc, F, gamma,
                        Ms_0, Ms_1, Ms_2, draws)
    avg_choice = choice[year_idx].mean(0)
    s_ca_cost.append(float(avg_choice[0]))
    s_bc_cost.append(float(avg_choice[1]))
    s_dc_cost.append(float(avg_choice[2]))
    s_cc_cost.append(float(avg_choice[3]))

s_ca_perc, s_bc_perc, s_dc_perc, s_cc_perc = [], [], [], []
for multipler in range(start, end, step):
    X_bc = X_bc_base.clone()
    # Adjust all 3 perception variables based on the multiplier
    X_bc[:,:3] = X_bc_base[:,:3].clone() * (1 + multipler/100)
    X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)
    choice, adopt = compute_avg_prob_util_ctf(X_ca_renorm, X_dc_renorm, X_cc_renorm, X_bc_renorm, mu, sd,
                        alpha_ca, alpha_dc, alpha_cc, alpha_bc, F, gamma,
                        Ms_0, Ms_1, Ms_2, draws)
    avg_choice = choice[year_idx].mean(0)
    s_ca_perc.append(float(avg_choice[0]))
    s_bc_perc.append(float(avg_choice[1]))
    s_dc_perc.append(float(avg_choice[2]))
    s_cc_perc.append(float(avg_choice[3]))

    # Plot and print the sensitivity of market share vs the changes in cost and perception.
gs = gridspec.GridSpec(1,2)
plt.figure(figsize=(16,6))
ax1 = plt.subplot(gs[0, 0])
plt.plot(np.array(range(start, end, step))/100, s_ca_cost, color='#E69F00', linewidth=5, label='Cash')
plt.plot(np.array(range(start, end, step))/100, s_bc_cost, color='#56B4E9', linewidth=5, label='CBDC')
plt.plot(np.array(range(start, end, step))/100, s_dc_cost, color='#009E73', linewidth=5, label='Debit')
plt.plot(np.array(range(start, end, step))/100, s_cc_cost, color='#CC79A7', linewidth=5, label='Credit')
plt.xlim([start/100, (end-step)/100])
plt.ylim([0, 1])
plt.xlabel('Increase in transaction cost')
plt.ylabel('Probability of using')


ax2 = plt.subplot(gs[0, 1], sharey=ax1)
plt.plot(np.array(range(start, end, step))/100, s_ca_perc, color='#E69F00', linewidth=5, label='Cash')
plt.plot(np.array(range(start, end, step))/100, s_bc_perc, color='#56B4E9', linewidth=5, label='CBDC')
plt.plot(np.array(range(start, end, step))/100, s_dc_perc, color='#009E73', linewidth=5, label='Debit')
plt.plot(np.array(range(start, end, step))/100, s_cc_perc, color='#CC79A7', linewidth=5, label='Credit')
plt.xlim([start/100, (end-step)/100])
plt.ylim([0, 1])
plt.xlabel('Increase in overall perception')
plt.legend()

plt.setp(ax2.get_yticklabels(), visible=False)
plt.tight_layout()
plt.show()

print('Steps')
print(np.array(range(start, end, step)))
print('Increase in cost')
print(np.array([s_ca_cost, s_bc_cost, s_dc_cost, s_cc_cost]).round(3))
print('Increase in perc')
print(np.array([s_ca_perc, s_bc_perc, s_dc_perc, s_cc_perc]).round(3))

### Utility contribution

### How does utility change in each situation
# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

alpha_bc = alpha_dc

# Compute the adoption utility WITHOUT CBDC
beta = draws[:, nid_index, :] * torch.exp(sd) * beta_mask + mu
delta_ca_nobc = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_dc_nobc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc_nobc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_nocbdc = stage2_bundle_emax(delta_ca_nobc, delta_dc_nobc, delta_cc_nobc, Ms_0, Ms_1, Ms_2)
emax_nocbdc = stage1_emax_ctf(F, gamma, stage2_emax_nocbdc)

# Compute the adoption utility WITH CBDC
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_cbdc = stage2_bundle_emax_ctf(delta_ca, delta_bc, delta_dc, delta_cc, Ms_0, Ms_1, Ms_2)
emax_cbdc = stage1_emax_ctf(F, gamma, stage2_emax_cbdc)

# Convert into dollar and put it in a table
expect_util_improvement = (emax_cbdc - emax_nocbdc).mean(0).squeeze()[nid[year_idx].unique().long()] / float(-mu[4])
util_improvement_table = data[['nid', 'empd', 'mard', 'agec', 'incc', 'gen',  'edu',
                 'urb', 'own', 'can', 'year', 'sph']].drop_duplicates()
util_improvement_table = util_improvement_table[util_improvement_table.year == 2017]
util_improvement_table['u'] = expect_util_improvement.detach().cpu()

# Group by and take mean of util improvement by demographics
util_improv_edu = util_improvement_table.groupby('edu').mean()[['u']]
util_improv_edu['eduname'] = ['High school', 'Grad School', 'Some uni.', '< high school', 'Tech. school', 'Uni degree']
util_improv_edu = util_improv_edu.sort_values(by='u', ascending=False)

util_improv_age = util_improvement_table.groupby('agec').mean()[['u']]
util_improv_age['agen'] = ['18-32', '33-42', '43-52', '53-61', '62-99']
util_improv_age = util_improv_age.sort_values(by='u', ascending=False)

util_improv_inc = util_improvement_table.groupby('incc').mean()[['u']]
util_improv_inc['incn'] = ['< 25k', '25k-45k', '45k-65k', '65k-90k', '> 90k']
util_improv_inc = util_improv_inc.sort_values(by='u', ascending=False)

mean_imp = util_improvement_table['u'].mean()

# plot the distribution and breakdown of util gain
gs = gridspec.GridSpec(2,3)
plt.figure(figsize=(16,12))

ax1 = plt.subplot(gs[0, :])
plt.hist(expect_util_improvement.detach().cpu(), bins=np.arange(float(expect_util_improvement.min().round()), float(expect_util_improvement.max().ceil()), 0.25), color='dimgrey')
plt.xlabel("Utility improvement in dollars")
plt.ylabel("Person count")

ax2 = plt.subplot(gs[1, 0])
plt.bar(x=util_improv_edu['eduname'], height=util_improv_edu['u'], align='center', color='dimgrey')
plt.ylabel("Utility improvement in dollars")
plt.xticks(rotation=90)

ax3 = plt.subplot(gs[1, 1], sharey=ax2)
plt.bar(x=util_improv_age['agen'], height=util_improv_age['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = plt.subplot(gs[1, 2], sharey=ax2)
plt.bar(x=util_improv_inc['incn'], height=util_improv_inc['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax4.get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()

# save to a file
util_improvement_table.to_csv('ecs3cl.csv', index=False)

# 6 bundles

# Compute the expected maximum in the 6 bundle case. Assume CBDC bundles have no extra cost or benefit
def stage1_emax_ctf_6(F, gamma, stage2_emax, bcost=0):
    fixed_cost = torch.cat((zero, F, zero, F))
    variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma, torch.zeros_like(cs), CSb * gamma), 1)
    util_after_cost = stage2_emax + fixed_cost + variable_cost
    return torch.logsumexp(util_after_cost, dim=2, keepdim=True)

# Compute the second stage expected maximum utility given acceptance rate of 6 bundles
def stage2_bundle_emax_ctf_6(delta_ca, delta_bc, delta_dc, delta_cc, \
                                             accept_prob_0, accept_prob_1, accept_prob_2, \
                                             accept_prob_3, accept_prob_4, accept_prob_5):

    u_0_base = torch.logsumexp(torch.cat([delta_ca], 2), dim=2, keepdim=True)
    u_1_base = torch.logsumexp(torch.cat([delta_ca, delta_dc], 2), dim=2, keepdim=True)
    u_2_base = torch.logsumexp(torch.cat([delta_ca, delta_dc, delta_cc], 2), dim=2, keepdim=True)
    u_3_base = torch.logsumexp(torch.cat([delta_ca, delta_bc], 2), dim=2, keepdim=True)
    u_4_base = torch.logsumexp(torch.cat([delta_ca, delta_bc, delta_dc], 2), dim=2, keepdim=True)
    u_5_base = torch.logsumexp(torch.cat([delta_ca, delta_bc, delta_dc, delta_cc], 2), dim=2, keepdim=True)

    u_0 = u_0_base
    u_1 = (accept_prob_0 + accept_prob_3) * u_0_base \
            + (accept_prob_1 + accept_prob_2 + accept_prob_4 + accept_prob_5) * u_1_base
    u_2 = (accept_prob_0 + accept_prob_3) * u_0_base \
            + (accept_prob_1 + accept_prob_4) * u_1_base \
            + (accept_prob_2 + accept_prob_5) * u_2_base

    # compute utility of CBDC bundles
    u_3 = (accept_prob_0 + accept_prob_1 + accept_prob_2) * u_0_base \
        + (accept_prob_3 + accept_prob_4 + accept_prob_5) * u_3_base
    u_4 = accept_prob_0 * u_0_base \
            + (accept_prob_1 + accept_prob_2) * u_1_base + accept_prob_3 * u_3_base \
            + (accept_prob_4 + accept_prob_5) * u_4_base
    u_5 = accept_prob_0 * u_0_base \
            + accept_prob_1 * u_1_base \
            + accept_prob_2 * u_2_base \
            + accept_prob_3 * u_3_base \
            + accept_prob_4 * u_4_base \
            + accept_prob_5 * u_5_base

    u = torch.cat((u_0, u_1, u_2, u_3, u_4, u_5), dim=2)
    utils = torch.matmul(u.transpose(1,2), nid_matrix.t()).transpose(1,2)
    return utils

# Compute the probability of adopting any of the 6 bundles
def adoption_prob_ctf_6(F, gamma, util, bcost=0):
    fixed_cost = torch.cat((zero, F, zero - bcost , F - bcost))
    variable_cost = torch.cat((torch.zeros_like(cs), CSb * gamma.t(), torch.zeros_like(cs), CSb * gamma.t()), 1)
    util_after_cost = util + fixed_cost + variable_cost
    likelihood = torch.nn.functional.softmax(util_after_cost, dim=2)
    return likelihood

# Compute the probability of using any of the payment methods given adoption and acceptance prob of 6 bundles
def usage_prob_ctf_6(delta_ca, delta_bc, delta_dc, delta_cc, \
                             adopt_prob_0, adopt_prob_1, adopt_prob_2,\
                             adopt_prob_3, adopt_prob_4, adopt_prob_5,\
                             accept_prob_0, accept_prob_1, accept_prob_2, \
                             accept_prob_3, accept_prob_4, accept_prob_5):
    p_0 = torch.nn.functional.softmax(torch.cat([delta_ca], dim=2), dim=2)
    p_1 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_dc], dim=2), dim=2)
    p_2 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_dc, delta_cc], dim=2), dim=2)
    p_3 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc], dim=2), dim=2)
    p_4 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc, delta_dc], dim=2), dim=2)
    p_5 = torch.nn.functional.softmax(torch.cat([delta_ca, delta_bc, delta_dc, delta_cc], dim=2), dim=2)

    p_ca_bundle0 = p_0[:,:,0:1]
    p_ca_bundle1 = p_1[:,:,0:1]
    p_dc_bundle1 = p_1[:,:,1:2]
    p_ca_bundle2 = p_2[:,:,0:1]
    p_dc_bundle2 = p_2[:,:,1:2]
    p_cc_bundle2 = p_2[:,:,2:3]

    p_ca_bundle3 = p_3[:,:,0:1]
    p_bc_bundle3 = p_3[:,:,1:2]
    p_ca_bundle4 = p_4[:,:,0:1]
    p_bc_bundle4 = p_4[:,:,1:2]
    p_dc_bundle4 = p_4[:,:,2:3]
    p_ca_bundle5 = p_5[:,:,0:1]
    p_bc_bundle5 = p_5[:,:,1:2]
    p_dc_bundle5 = p_5[:,:,2:3]
    p_cc_bundle5 = p_5[:,:,3:4]

    p_ca = p_ca_bundle0 * adopt_prob_0 * 1 \
        + p_ca_bundle0 * (adopt_prob_1 + adopt_prob_2 + adopt_prob_3 + adopt_prob_4 + adopt_prob_5) * accept_prob_0 \
        + p_ca_bundle0 * adopt_prob_3 * (accept_prob_1 + accept_prob_2) \
        + p_ca_bundle0 * (adopt_prob_1 + adopt_prob_2) * accept_prob_3 \
        + p_ca_bundle1 * adopt_prob_1 * (accept_prob_1 + accept_prob_2 + accept_prob_4 + accept_prob_5) \
        + p_ca_bundle1 * (adopt_prob_2 + adopt_prob_4 + adopt_prob_5) * accept_prob_1 \
        + p_ca_bundle1 * adopt_prob_4 * accept_prob_2 \
        + p_ca_bundle1 * adopt_prob_2 * accept_prob_4 \
        + p_ca_bundle2 * adopt_prob_2 * (accept_prob_2 + accept_prob_5) \
        + p_ca_bundle2 * adopt_prob_5 * accept_prob_2 \
        + p_ca_bundle3 * adopt_prob_3 * (accept_prob_3 + accept_prob_4 + accept_prob_5) \
        + p_ca_bundle3 * (adopt_prob_4 + adopt_prob_5) * accept_prob_3 \
        + p_ca_bundle4 * (adopt_prob_4 + adopt_prob_5) * accept_prob_4 \
        + p_ca_bundle4 * adopt_prob_4 * accept_prob_5 \
        + p_ca_bundle5 * adopt_prob_5 * accept_prob_5

    p_bc = p_bc_bundle3 * (adopt_prob_3 + adopt_prob_4 + adopt_prob_5) * accept_prob_3 \
        + p_bc_bundle3 * adopt_prob_3 * (accept_prob_4 + accept_prob_5) \
        + p_bc_bundle4 * (adopt_prob_4 + adopt_prob_5) * accept_prob_4 \
        + p_bc_bundle4 * adopt_prob_4 * accept_prob_5 \
        + p_bc_bundle5 * adopt_prob_5 * accept_prob_5

    p_dc = p_dc_bundle1 * (adopt_prob_1 + adopt_prob_2 + adopt_prob_4 + adopt_prob_5) * accept_prob_1 \
        + p_dc_bundle1 * adopt_prob_1 * (accept_prob_2 + accept_prob_4 + accept_prob_5) \
        + p_dc_bundle1 * adopt_prob_4 * accept_prob_2 \
        + p_dc_bundle1 * adopt_prob_2 * accept_prob_4 \
        + p_dc_bundle2 * adopt_prob_2 * (accept_prob_2 + accept_prob_5) \
        + p_dc_bundle2 * adopt_prob_5 * accept_prob_2 \
        + p_dc_bundle4 * (adopt_prob_4 + adopt_prob_5) * accept_prob_4 \
        + p_dc_bundle4 * adopt_prob_4 * accept_prob_5 \
        + p_dc_bundle5 * adopt_prob_5 * accept_prob_5

    p_cc = p_cc_bundle2 * (adopt_prob_2 + adopt_prob_5) * accept_prob_2 \
        + p_cc_bundle2 * adopt_prob_2 * accept_prob_5 \
        + p_cc_bundle5 * adopt_prob_5 * accept_prob_5
    return torch.cat((p_ca, p_bc, p_dc, p_cc), dim=2)

# util improvement for 6 bundles

mu, sd, alpha_ca, alpha_dc,\
    alpha_cc, F, gamma, draws, l = get_params_from('spec3_mix_cs.npy')

# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

alpha_bc = alpha_dc

X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

# Assume 75% acceptance rate of CBDC
pickup_rate = 0.75
accept_prob_0 = Ms_0 * (1 - pickup_rate)
accept_prob_1 = Ms_1 * (1 - pickup_rate)
accept_prob_2 = Ms_2 * (1 - pickup_rate)
accept_prob_3 = Ms_0 * pickup_rate
accept_prob_4 = Ms_1 * pickup_rate
accept_prob_5 = Ms_2 * pickup_rate

# Compute the adoption utility WITHOUT CBDC (infinite punishment for CBDC)
beta = draws[:, nid_index, :] * torch.exp(sd) * beta_mask + mu
delta_ca_nobc = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc_nobc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True) - 999999999999
delta_dc_nobc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc_nobc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_nocbdc = stage2_bundle_emax_ctf_6(delta_ca_nobc, delta_bc_nobc, delta_dc_nobc, delta_cc_nobc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
emax_nocbdc = stage1_emax_ctf_6(F, gamma, stage2_emax_nocbdc, bcost=0)

# Compute the adoption utility WITH CBDC
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_cbdc = stage2_bundle_emax_ctf_6(delta_ca, delta_bc, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
emax_cbdc = stage1_emax_ctf_6(F, gamma, stage2_emax_cbdc, bcost=0)

# Convert into dollar and put it in a table
expect_util_improvement = (emax_cbdc - emax_nocbdc).mean(0).squeeze()[nid[year_idx].unique().long()] / float(-mu[4])
util_improvement_table = data[['nid', 'empd', 'mard', 'agec', 'incc', 'gen',  'edu',
                 'urb', 'own', 'can', 'year', 'sph']].drop_duplicates()
util_improvement_table = util_improvement_table[util_improvement_table.year == 2017]
util_improvement_table['u'] = expect_util_improvement.detach().cpu()

# Group by and take mean of util improvement by demographics
util_improv_edu = util_improvement_table.groupby('edu').mean()[['u']]
util_improv_edu['eduname'] = ['High school', 'Grad School', 'Some uni.', '< high school', 'Tech. school', 'Uni degree']
util_improv_edu = util_improv_edu.sort_values(by='u', ascending=False)

util_improv_age = util_improvement_table.groupby('agec').mean()[['u']]
util_improv_age['agen'] = ['18-32', '33-42', '43-52', '53-61', '62-99']
util_improv_age = util_improv_age.sort_values(by='u', ascending=False)

util_improv_inc = util_improvement_table.groupby('incc').mean()[['u']]
util_improv_inc['incn'] = ['< 25k', '25k-45k', '45k-65k', '65k-90k', '> 90k']
util_improv_inc = util_improv_inc.sort_values(by='u', ascending=False)

mean_imp = util_improvement_table['u'].mean()

# plot the distribution and breakdown of util gain\
gs = gridspec.GridSpec(2,3)
plt.figure(figsize=(16,12))

ax1 = plt.subplot(gs[0, :])
plt.hist(expect_util_improvement.detach().cpu(), bins=np.arange(float(expect_util_improvement.min().round()), float(expect_util_improvement.max().ceil()), 0.2), color='dimgrey')
plt.xlabel("Utility improvement in dollars")
plt.ylabel("Person count")

ax2 = plt.subplot(gs[1, 0])
plt.bar(x=util_improv_edu['eduname'], height=util_improv_edu['u'], align='center', color='dimgrey')
plt.ylabel("Utility improvement in dollars")
plt.xticks(rotation=90)

ax3 = plt.subplot(gs[1, 1], sharey=ax2)
plt.bar(x=util_improv_age['agen'], height=util_improv_age['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = plt.subplot(gs[1, 2], sharey=ax2)
plt.bar(x=util_improv_inc['incn'], height=util_improv_inc['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax4.get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()

# save to a file
util_improvement_table.to_csv('ecs6bb.csv', index=False)

mean_imp

### 6 Bundle but CBDC is the default now

# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

alpha_bc = alpha_dc

X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

# Assume 75% acceptance rate of CBDC
pickup_rate = 0.75
accept_prob_0 = Ms_0 * (1 - pickup_rate)
accept_prob_1 = Ms_1 * (1 - pickup_rate)
accept_prob_2 = Ms_2 * (1 - pickup_rate)
accept_prob_3 = Ms_0 * pickup_rate
accept_prob_4 = Ms_1 * pickup_rate
accept_prob_5 = Ms_2 * pickup_rate

# Compute the adoption utility WITHOUT CBDC (infinite punishment for CBDC)
beta = draws[:, nid_index, :] * torch.exp(sd) * beta_mask + mu
delta_ca_nobc = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc_nobc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True) - 999999999999
delta_dc_nobc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc_nobc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_nocbdc = stage2_bundle_emax_ctf_6(delta_ca_nobc, delta_bc_nobc, delta_dc_nobc, delta_cc_nobc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
emax_nocbdc = stage1_emax_ctf_6(F, gamma, stage2_emax_nocbdc, bcost=0)

# Compute the adoption utility WITH CBDC
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)
stage2_emax_cbdc = stage2_bundle_emax_ctf_6(delta_bc, delta_ca, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
emax_cbdc = stage1_emax_ctf_6(F, gamma, stage2_emax_cbdc, bcost=0)

# Convert into dollar and put it in a table
expect_util_improvement = (emax_cbdc - emax_nocbdc).mean(0).squeeze()[nid[year_idx].unique().long()] / float(-mu[4])
util_improvement_table = data[['nid', 'empd', 'mard', 'agec', 'incc', 'gen',  'edu',
                 'urb', 'own', 'can', 'year', 'sph']].drop_duplicates()
util_improvement_table = util_improvement_table[util_improvement_table.year == 2017]
util_improvement_table['u'] = expect_util_improvement.detach().cpu()

# Group by and take mean of util improvement by demographics
util_improv_edu = util_improvement_table.groupby('edu').mean()[['u']]
util_improv_edu['eduname'] = ['High school', 'Grad School', 'Some uni.', '< high school', 'Tech. school', 'Uni degree']
util_improv_edu = util_improv_edu.sort_values(by='u', ascending=False)

util_improv_age = util_improvement_table.groupby('agec').mean()[['u']]
util_improv_age['agen'] = ['18-32', '33-42', '43-52', '53-61', '62-99']
util_improv_age = util_improv_age.sort_values(by='u', ascending=False)

util_improv_inc = util_improvement_table.groupby('incc').mean()[['u']]
util_improv_inc['incn'] = ['< 25k', '25k-45k', '45k-65k', '65k-90k', '> 90k']
util_improv_inc = util_improv_inc.sort_values(by='u', ascending=False)

mean_imp = util_improvement_table['u'].mean()

# plot the distribution and breakdown of util gain\
gs = gridspec.GridSpec(2,3)
plt.figure(figsize=(16,12))

ax1 = plt.subplot(gs[0, :])
plt.hist(expect_util_improvement.detach().cpu(), bins=np.arange(float(expect_util_improvement.min().round()), float(expect_util_improvement.max().ceil()), 0.2), color='dimgrey')
plt.xlabel("Utility improvement in dollars")
plt.ylabel("Person count")

ax2 = plt.subplot(gs[1, 0])
plt.bar(x=util_improv_edu['eduname'], height=util_improv_edu['u'], align='center', color='dimgrey')
plt.ylabel("Utility improvement in dollars")
plt.xticks(rotation=90)

ax3 = plt.subplot(gs[1, 1], sharey=ax2)
plt.bar(x=util_improv_age['agen'], height=util_improv_age['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = plt.subplot(gs[1, 2], sharey=ax2)
plt.bar(x=util_improv_inc['incn'], height=util_improv_inc['u'], align='center', color='dimgrey')
plt.xticks(rotation=90)
plt.setp(ax4.get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()

# save to a file
util_improvement_table.to_csv('ecs6bb.csv', index=False)

mean_imp

### 6 bundle Pickup rate vs adoption and usage

# We are going to vary the acceptance rate of CBDC and see how the usage and acceptance probability changes with it.

# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

alpha_bc = alpha_dc

X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

multiplier_range = np.array(range(0, 101, 10))

beta = draws[:, nid_index, :] * torch.exp(sd) + mu
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)

adopt_probs_t = []
usage_probs_t = []
for multiplier in multiplier_range:
    # Assume Bernoulli and bifurcates store acceptance between CBDC and no-CBDC
    pickup_rate = multiplier/100
    accept_prob_0 = Ms_0 * (1 - pickup_rate)
    accept_prob_1 = Ms_1 * (1 - pickup_rate)
    accept_prob_2 = Ms_2 * (1 - pickup_rate)
    accept_prob_3 = Ms_0 * pickup_rate
    accept_prob_4 = Ms_1 * pickup_rate
    accept_prob_5 = Ms_2 * pickup_rate

    # Compute the adopt and usage rate
    util = stage2_bundle_emax_ctf_6(delta_ca, delta_bc, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
    short_adopt = adoption_prob_ctf_6(F, gamma, util, bcost=0)
    adopt_prob_tm = short_adopt[:, nid_index.long(), :]
    usage_prob_tm = usage_prob_ctf_6(delta_ca, delta_bc, delta_dc, delta_cc, \
                   adopt_prob_tm[:, :, 0:1], adopt_prob_tm[:, :, 1:2], adopt_prob_tm[:, :, 2:3], adopt_prob_tm[:, :, 3:4], adopt_prob_tm[:, :, 4:5], adopt_prob_tm[:, :, 5:6], \
                   accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
    adopt_probs_t.append(short_adopt.mean(0)[nid[year_idx].unique().long()].mean(0))
    usage_probs_t.append(usage_prob_tm.mean(0)[year_idx].mean(0))
adopt_prob = torch.stack(adopt_probs_t).detach().cpu().numpy()
usage_prob = torch.stack(usage_probs_t).detach().cpu().numpy()

print('Usage prob\n', usage_prob.T.round(2))
print('Adopt prob\n', adopt_prob.T.round(2))

# Plot it as a function of the multiplier
gs = gridspec.GridSpec(2,1)
plt.figure(figsize=(16,12))
ax1 = plt.subplot(gs[0, 0])
plt.plot(multiplier_range, usage_prob[:,0], color='#E69F00', linewidth=5, label='Cash')
plt.plot(multiplier_range, usage_prob[:,1], color='#56B4E9', linewidth=5, label='CBDC')
plt.plot(multiplier_range, usage_prob[:,2], color='#009E73', linewidth=5, label='Debit')
plt.plot(multiplier_range, usage_prob[:,3], color='#CC79A7', linewidth=5, label='Credit')
plt.legend()
plt.xticks(range(0,101, 10))
plt.xlim([0,100])
plt.ylim([0,0.5])
plt.ylabel('Usage rate')

ax2 = plt.subplot(gs[1, 0], sharex=ax1)
plt.plot(multiplier_range, adopt_prob[:,0:3].sum(1), color='darkorange', linewidth=5, label='Do not adopt CBDC')
plt.plot(multiplier_range, adopt_prob[:,3:6].sum(1), color='navy', linewidth=5, label='Adopt CBDC')
plt.legend()
plt.xlabel('CBDC Merchant pickup rate (%)')
plt.ylim([0,1])
plt.ylabel('CBDC adoption rate')
plt.setp(ax1.get_xticklabels(), visible=False)

plt.tight_layout()
plt.show()

### 6 bundle Pickup rate vs adoption and usage (CBDC is default)

#We are going to vary the acceptance rate of CBDC and see how the usage and acceptance probability changes with it.

# Cash like
# X_bc = X_ca.clone()

# Debit like
# X_bc = X_dc.clone()

# Best of both
X_bc = torch.max(X_ca, X_dc).clone()
X_bc[:,4] = X_dc[:,4].clone()

alpha_bc = alpha_dc

X_ca_renorm, X_bc_renorm, X_dc_renorm, X_cc_renorm = renormalize(X_bc)

multiplier_range = np.array(range(0, 101, 10))

beta = draws[:, nid_index, :] * torch.exp(sd) + mu
delta_ca = (beta * X_ca_renorm).sum(2, keepdim=True) + (Zbj * alpha_ca).sum(1, keepdim=True)
delta_bc = (beta * X_bc_renorm).sum(2, keepdim=True) + (Zbj * alpha_bc).sum(1, keepdim=True)
delta_dc = (beta * X_dc_renorm).sum(2, keepdim=True) + (Zbj * alpha_dc).sum(1, keepdim=True)
delta_cc = (beta * X_cc_renorm).sum(2, keepdim=True) + (Zbj * alpha_cc).sum(1, keepdim=True)

adopt_probs_t = []
usage_probs_t = []
for multiplier in multiplier_range:
    # Assume Bernoulli and bifurcates store acceptance between CBDC and no-CBDC
    pickup_rate = multiplier/100
    accept_prob_0 = Ms_0 * (1 - pickup_rate)
    accept_prob_1 = Ms_1 * (1 - pickup_rate)
    accept_prob_2 = Ms_2 * (1 - pickup_rate)
    accept_prob_3 = Ms_0 * pickup_rate
    accept_prob_4 = Ms_1 * pickup_rate
    accept_prob_5 = Ms_2 * pickup_rate

    # Compute the adopt and usage rate
    util = stage2_bundle_emax_ctf_6(delta_bc, delta_ca, delta_dc, delta_cc, accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
    short_adopt = adoption_prob_ctf_6(F, gamma, util, bcost=0)
    adopt_prob_tm = short_adopt[:, nid_index.long(), :]
    usage_prob_tm = usage_prob_ctf_6(delta_bc, delta_ca, delta_dc, delta_cc, \
                   adopt_prob_tm[:, :, 0:1], adopt_prob_tm[:, :, 1:2], adopt_prob_tm[:, :, 2:3], adopt_prob_tm[:, :, 3:4], adopt_prob_tm[:, :, 4:5], adopt_prob_tm[:, :, 5:6], \
                   accept_prob_0, accept_prob_1, accept_prob_2, accept_prob_3, accept_prob_4, accept_prob_5)
    adopt_probs_t.append(short_adopt.mean(0)[nid[year_idx].unique().long()].mean(0))
    usage_probs_t.append(usage_prob_tm.mean(0)[year_idx].mean(0))
adopt_prob = torch.stack(adopt_probs_t).detach().cpu().numpy()
usage_prob = torch.stack(usage_probs_t).detach().cpu().numpy()

print('Usage prob\n', usage_prob.T.round(2))
print('Adopt prob\n', adopt_prob.T.round(2))

# Plot it as a function of the multiplier
gs = gridspec.GridSpec(2,1)
plt.figure(figsize=(16,12))
ax1 = plt.subplot(gs[0, 0])
plt.plot(multiplier_range, usage_prob[:,1], color='#E69F00', linewidth=5, label='Cash')
plt.plot(multiplier_range, usage_prob[:,0], color='#56B4E9', linewidth=5, label='CBDC')
plt.plot(multiplier_range, usage_prob[:,2], color='#009E73', linewidth=5, label='Debit')
plt.plot(multiplier_range, usage_prob[:,3], color='#CC79A7', linewidth=5, label='Credit')
plt.legend()
plt.xticks(range(0,101, 10))
plt.xlim([0,100])
plt.ylim([0,0.5])
plt.ylabel('Usage rate')

ax2 = plt.subplot(gs[1, 0], sharex=ax1)
plt.plot(multiplier_range, adopt_prob[:,0:3].sum(1), color='darkorange', linewidth=5, label='Do not adopt cash')
plt.plot(multiplier_range, adopt_prob[:,3:6].sum(1), color='navy', linewidth=5, label='Adopt cash')
plt.legend()
plt.xlabel('Cash Merchant pickup rate (%)')
plt.ylim([0,1])
plt.ylabel('Cash adoption rate')
plt.setp(ax1.get_xticklabels(), visible=False)

plt.tight_layout()
plt.show()