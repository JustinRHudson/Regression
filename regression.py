import numpy as np
import time

# Let me define my regression class
class regression():
    '''
        Custom regression class. This takes in a model, predictors, and a predictand and
        uses those to perform the desired regression. Uses MSE as a loss function and
        gradient descent as my methodology to find the ideal parameters.
    '''
    def __init__(self,Y:np.ndarray,X:np.ndarray,learning_rate:float = 0.01,max_iterations:int = 10000,
                 model_type:str = 'linear',random_seed:int = None,patience_pct:float = 0.01,
                 predictor_names:list = None) -> None:
        '''
            Y (np.ndarray): The predictand. This should be a 1-D Numpy Array of length N.
            X (np.ndarray): The predictors. This should be a M x N 2-D numpy array.
            learning_rate (float): How large of a step should be taken between gradient descent steps. Default 0.01.
            max_iterations (int): How many interations should be done at most to find the optimal coefficients. Default 10000.
            model_type (str): Either 'linear' or 'linear_with_interactions' determines the type of model generated. Default 'linear'
            random_seed (int): The preferred random seed if there is none, default None.
            patience_pct (float): How small a change should be tolerated over 10 iterations before the fitting will end early.
            predictor_names (np.ndarray): The names of the predictors. Used to report summary statistics. Default None.
        '''
        self.Y = Y #the predictand
        self.X = X #the predictors

        self.predictor_names = predictor_names
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.patience_pct = patience_pct
        self.model_type = model_type
        self.select_model()
        self.random_seed = random_seed
        self.set_seed()
        self.fit_history = []
        self.weight_history = []
        self.best_weights = [-9999]
        self.best_epoch = -9999
        self.num_epochs = 0
    
    def set_seed(self):
        if not self.random_seed == None:
            np.random.seed(self.random_seed)
        else:
            np.random.seed(int(time.time() / 2 - 144))
    
    def calculate_MSE(self):
        if len(self.X.shape) == 1:
            MSE = (1.0/len(self.Y)) * np.sum((self.weights[:-1]*self.X + self.weights[-1] - self.Y)**2)
        else:
            weighted_xs = np.array([self.weights[i]*self.X[i] for i in range(self.X.shape[0])])
            MSE = (1.0/len(self.Y)) * np.sum((np.sum(weighted_xs,axis = 0) + self.weights[-1] - self.Y)**2)

        return MSE
    
    def kronecker_delta(self,i,j):
        if i == j:
            return 1
        else:
            return 0
    
    def custom_loss(self):
        if len(self.X.shape) == 1:
            loss = (1.0/len(self.Y)) * np.sum((self.weights[0]*self.X - (self.Y - self.weights[-1]))**2)
            return loss
        else:
            total_loss = 0
            for i in range(self.X.shape[0]):
                weighted_xs = np.sum(np.array([(1-self.kronecker_delta(i,j))*self.weights[j]*self.X[j] for j in range(self.X.shape[0])]),axis = 0)
                total_loss += np.sum((self.weights[i]*self.X[i] - (self.Y - self.weights[-1] - weighted_xs))**2)
            total_loss /= len(self.Y)
            
            return total_loss


    def initialize_weights(self,num_weights:int):
        '''
            Initializes the weights of the model based off the type of model and
            a random seed if that is specififed. The last weight in the array is
            the weight corresponding to the Y-intercept
        '''
        self.weights = np.random.random(num_weights) * 2
    
    def update_weight_history(self):
        weight_array = np.array([self.weights[i] for i in range(len(self.weights))])
        self.weight_history.append(weight_array)

    
    def select_model(self):
        model_dict = {'linear':self.linear_model,'linear_with_interaction':self.linear_interaction_model}
        if self.model_type in model_dict.keys():
            self.model = model_dict[self.model_type]
        else:
            raise ValueError("Invalid Model Type Entered. Options are linear and linear_with_interaction.")
    
    def linear_model(self):
        #first define the number of weights needed
        if len(self.X.shape) == 2:
            num_weights = self.X.shape[0]+1
        else:
            num_weights = 2
        self.initialize_weights(num_weights)

        return None
    
    def linear_model_gradient_descent(self):
        #use gradient descent to get the new weights
        if len(self.X.shape) == 1:
            weight_gradients = (2.0 / len(self.Y)) * np.sum((self.X*self.weights[:-1] + self.weights[-1] - self.Y) * self.X)
            y_int_gradient = (2.0 / len(self.Y)) * np.sum((self.X*self.weights[:-1] + self.weights[-1] - self.Y))
            self.weights[:-1] = self.weights[:-1] - self.learning_rate * weight_gradients
            self.weights[-1] = self.weights[-1] - self.learning_rate * y_int_gradient
        else:
            weight_gradients = np.empty(self.X.shape[0])
            for i in range(len(weight_gradients)):
                weighted_xs = np.array([np.array(self.weights[i]*self.X[i]) for i in range(self.X.shape[0])])
                weight_gradients[i] = (2.0 / len(self.Y)) * np.sum(( (np.sum(weighted_xs,axis=0) + self.weights[-1] - self.Y) * self.X[i]))
            y_int_gradient = (2.0 / len(self.Y)) * np.sum(np.sum(weighted_xs,axis = 0) + self.weights[-1] - self.Y)
            for i in range(len(weight_gradients)):
                self.weights[i] = self.weights[i] - self.learning_rate * weight_gradients[i]
            self.weights[-1] = self.weights[-1] - self.learning_rate * y_int_gradient


        return None

    def fit_linear_model(self):
        #initialize the model
        self.linear_model()
        for i in range(self.max_iterations):
            self.num_epochs += 1
            self.linear_model_gradient_descent()
            # Add the new weights to the history
            self.update_weight_history()
            # get the MSE for this iterations
            self.fit_history.append(self.calculate_MSE())
            # check if the change in MSE is sufficiently large to justify continuing
            # to fit over the past 10 iterations
            if i % 10 == 0 and i > 0:
                current_MSE = self.fit_history[-1]
                past_10_MSEs = np.array(self.fit_history[-11:-1])
                MSE_pcts = current_MSE / past_10_MSEs
                if np.abs(np.mean(MSE_pcts) - 1) < self.patience_pct:
                    # if the change is sufficently small take the best weights
                    best_mse_ind = np.where(np.array(self.fit_history) == np.nanmin(np.array(self.fit_history)))[0][0]
                    self.best_epoch = best_mse_ind
                    self.best_weights = self.weight_history[best_mse_ind]
                    break
            else:
                best_mse_ind = np.where(np.array(self.fit_history) == np.nanmin(np.array(self.fit_history)))[0][0]
                self.best_epoch = best_mse_ind
                self.best_weights = self.weight_history[best_mse_ind]
        
    def linear_interaction_model(self):
        if len(self.X.shape) == 2:
            num_weights = int(self.X.shape[0] + 1 + (np.math.factorial(self.X.shape[0])) / (2*np.math.factorial(self.X.shape[0]-2)))
            self.initialize_weights(num_weights)
        else:
            raise ValueError("Cannot Do Interactions with only 1 Predictand.\nReverting to Linear Model.")
        
        return None
    
    def get_interaction_pair_indices(self):
        max_ind = self.X.shape[0]
        pairs = []
        for i in range(max_ind):
            for j in range(i+1,max_ind):
                pairs.append(np.array([i,j]))
        return np.array(pairs)


    def interaction_y_pred(self):
        #calculate Y_pred
        Y_pred = np.zeros(len(self.Y))
        # first add in what you get from the non-interacting Xs
        for i in range(self.X.shape[0]):
            Y_pred += self.weights[i] * self.X[i]
        # next add in what you get from the interacting Xs
        int_pair_inds = self.get_interaction_pair_indices()
        for i in range(len(int_pair_inds)):
            Y_pred += self.weights[i+self.X.shape[0]] * self.X[int_pair_inds[i][0]] * self.X[int_pair_inds[i][1]]
        Y_pred += self.weights[-1]

        return Y_pred


    def interaction_mse(self):

        Y_pred = self.interaction_y_pred()
        MSE = (1.0 / len(self.Y)) * np.sum((Y_pred - self.Y)**2)
        
        return MSE
    
    def interaction_model_gradient_descent(self):
        weight_gradients = np.empty(len(self.weights))
        Y_pred = self.interaction_y_pred()
        int_pair_inds = self.get_interaction_pair_indices()
        for i in range(self.X.shape[0]):
            weight_gradients[i] = (2.0 / len(self.Y)) * np.sum((Y_pred - self.Y) * self.X[i])
        for i in range(len(int_pair_inds)):
            weight_gradients[i+self.X.shape[0]] = (2.0 / len(self.Y)) * np.sum((Y_pred - self.Y)*self.X[int_pair_inds[i][0]]*self.X[int_pair_inds[i][1]])
        y_int_gradient = (2.0 / len(self.Y)) * np.sum(Y_pred - self.Y)
        #now update the weights with the gradients
        for i in range(len(self.weights)-1):
            self.weights[i] = self.weights[i] - self.learning_rate * weight_gradients[i]
        self.weights[-1] = self.weights[-1] - self.learning_rate * y_int_gradient
        return None

    def fit_interaction_model(self):
        #initialize the model
        self.linear_interaction_model()
        for i in range(self.max_iterations):
            self.num_epochs += 1
            self.interaction_model_gradient_descent()
            # Add the new weights to the history
            self.update_weight_history()
            # get the MSE for this iterations
            self.fit_history.append(self.interaction_mse())
            # check if the change in MSE is sufficiently large to justify continuing
            # to fit over the past 10 iterations
            if i % 10 == 0 and i > 0:
                current_MSE = self.fit_history[-1]
                past_10_MSEs = np.array(self.fit_history[-11:-1])
                MSE_pcts = current_MSE / past_10_MSEs
                if np.abs(np.mean(MSE_pcts) - 1) < self.patience_pct:
                    # if the change is sufficently small take the best weights
                    best_mse_ind = np.where(np.array(self.fit_history) == np.nanmin(np.array(self.fit_history)))[0][0]
                    self.best_epoch = best_mse_ind
                    self.best_weights = self.weight_history[best_mse_ind]
                    break
            else:
                best_mse_ind = np.where(np.array(self.fit_history) == np.nanmin(np.array(self.fit_history)))[0][0]
                self.best_epoch = best_mse_ind
                self.best_weights = self.weight_history[best_mse_ind]
        

    def make_prediction(self,pred_X):
        '''
            Given an already fit model generates a time series given a set of inputs.        
        '''
        if len(pred_X) != self.X.shape[0]:
            raise ValueError("Inputs are of Wrong Shape for Prediction.")

        if self.model_type == 'linear':
            prediction = np.dot(self.best_weights[:-1],pred_X) + self.best_weights[-1]

            return prediction
        elif self.model_type == 'linear_with_interaction':
            int_pred_pairs = self.get_interaction_pair_indices()
            prediction = np.dot(self.best_weights[:self.X.shape[0]],pred_X) + self.best_weights[-1]
            for i in range(len(int_pred_pairs)):
                prediction += self.best_weights[i+self.X.shape[0]] * pred_X[int_pred_pairs[i][0]] * pred_X[int_pred_pairs[i][1]]
            
            return prediction
        else:
            raise TypeError("Invalid Model Type Assigned to current model")

    def report_model_skill(self):
        '''Returns the r-squared and MSE of the model's regression onto the 
        Y data it was trained on.
        
        Returns (r^2, mse)'''

        model_prediction = self.make_prediction(self.X)
        r_squared = np.corrcoef(self.Y,model_prediction)[0,1]**2
        mse = self.calculate_MSE()

        return r_squared,mse
    

    def linear_model_weight_reporter(self):
        '''
            Reports the model weights for a linear model.
        '''

        if self.model_type != 'linear':
            return -9999
    
        #check whether or not predictor names were given
        if type(self.predictor_names) != list:
            self.predictor_names = ['Variable ' + str(int(i+1)) for i in range(self.X.shape[0])]
        
        #Get the sorted indices of the best weights and report them (excluding the bias term).
        weights_to_sort = self.best_weights[:-1]
        sorted_inds = np.flip(np.argsort(weights_to_sort))

        #now make a numpy array that is the sorted weights with the names paired
        sorted_names = [self.predictor_names[si] for si in sorted_inds]
        sorted_weights = [weights_to_sort[si] for si in sorted_inds]

        sorted_names.append('Bias Term')
        sorted_weights.append(self.best_weights[-1])

        return sorted_names,sorted_weights
    
    def linear_interaction_weight_reporter(self):
        '''
            Reports the model weights for a linear interaction model.
        '''

        if self.model_type != 'linear_with_interaction':
            return -9999
        
        #check whether or not predictor names were given
        if type(self.predictor_names) != list:
            self.predictor_names = ['Variable ' + str(int(i+1)) for i in range(self.X.shape[0])]
        
        #now add the interaction variables to the predictor names
        #first get the interaction pair inds
        int_pair_inds = self.get_interaction_pair_indices()
        #now use these to fill out the remaining var names
        for i in range(len(int_pair_inds)):
            str_to_append = f'{self.predictor_names[int_pair_inds[i][0]]}-{self.predictor_names[int_pair_inds[i][1]]} Interaction'
            self.predictor_names.append(str_to_append)
        
        #Get the sorted indices of the best weights and report them (excluding the bias term).
        weights_to_sort = self.best_weights[:-1]
        sorted_inds = np.flip(np.argsort(weights_to_sort))

        sorted_names = [self.predictor_names[si] for si in sorted_inds]
        sorted_weights = [weights_to_sort[si] for si in sorted_inds]

        sorted_names.append('Bias Term')
        sorted_weights.append(self.best_weights[-1])

        return sorted_names,sorted_weights

    def summary_report(self):
        '''
            For an already fit model reports each predictor and weight in a sorted table,
            to provide information and guidance to the user.
        '''
        if self.model_type == 'linear':
            names,weights = self.linear_model_weight_reporter()
        elif self.model_type == 'linear_with_interaction':
            names,weights = self.linear_interaction_weight_reporter()
        
        print('Model Weight Summary Statistics:')
        print('--------------------------------')
        for M in range(len(names)):
            print(names[M] + ': ' + f'{weights[M]:0.3f}')
        print('--------------------------------')
        
        return None


    def fit(self):
        start_time = time.time()
        if self.model_type == 'linear':
            self.fit_linear_model()
        elif self.model_type == 'linear_with_interaction':
            self.fit_interaction_model()
        end_time = time.time()
        model_r2,model_mse = self.report_model_skill()
        print(f"Fit Complete, Elapsed Time (Seconds): {end_time - start_time:.2f}")
        print(f"Epochs Needed: {self.num_epochs}, {(self.num_epochs/self.max_iterations)*100:.1f}% of Max")
        print(f"Model R^2: {model_r2:.03f}, Model MSE: {model_mse:.03f}")

        return None
    
    def feature_importance(self):
        '''
            Gauges the importances of features using permutation feature importance.
        '''

        return None
    

        
        