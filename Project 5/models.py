import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # return the dot product of weights and input vector.
        return nn.DotProduct(self.get_weights(), x)
        
        
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # calculate the prediction output and return 1 if output is greater than 0 else return -1.
        predict_output = nn.as_scalar(self.run(x))
        if predict_output >=0:
            return 1
        else:
            return -1
        
        
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Iterate for 500 epochs
        for epoch in range(500):
            
            # Iterate through dataset and get the x and y coordinates.
            for xval, yval in dataset.iterate_once(1):
                
                # get the prediction value for input x
                predict_output = self.get_prediction(xval)
                
                # validate the predicted output with actual value, if not equal update the weights.
                if predict_output != nn.as_scalar(yval):
                    self.get_weights().update(xval, nn.as_scalar(yval))
        
    
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        # Initialize, batch size, feature layers, weight and bias vectors.
        self.batch_size = 20
        self.feature_layer1 = 200
        self.feature_layer2 = 1
        self.feature_layer3 = 1
        
        # weight vector1 of size (1, feature layer1)
        self.weight1 = nn.Parameter(1, self.feature_layer1)
        
        # weight vector2 of size (feature layer1, feature layer2)
        self.weight2 = nn.Parameter(self.feature_layer1, self.feature_layer2)
        
        # weight vector1 of size (feature layer2, feature layer3)
        self.weight3 = nn.Parameter(self.feature_layer2,self.feature_layer3)
        
        # bias vector1 of size (1, feature layer1)
        self.bias1= nn.Parameter(1,self.feature_layer1)
        
        # bias vector1 of size (1, feature layer2)
        self.bias2 = nn.Parameter(1,self.feature_layer2)
        
        # bias vector1 of size (1, feature layer3)
        self.bias3 = nn.Parameter(1,self.feature_layer3)
        
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        
        feature1 = nn.Linear(x, self.weight1)   # add linear layer as initial layer with weight
        layer1 = nn.AddBias(feature1,self.bias1) # add bias to the feature.
        layer1 = nn.ReLU(layer1)    # Add ReLU activation layer to input layer.
        feature2 = nn.Linear(layer1, self.weight2)  # add linear layer to as second layer with weight.
        layer2 = nn.AddBias(feature2, self.bias2)   # add bias to the feature.
        
        return layer2 # return layer2 which is series of added layers.

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # predict y for the input x by calling RUN function.
        predicted_y = self.run(x)
        
        # calculate squared loss with respect to actual y and predicted y.
        loss = nn.SquareLoss(predicted_y, y)
        
        return loss   # return loss
        
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # initialize learning rate and number of iteration, total loss and total samples.
        learning_rate = -0.006
        num_iteration = 1
        total_loss = 0
        total_samples = 0
        
        # iterate through x, y coordinates in the dataset w.r.t batch size.
        for x,y in dataset.iterate_forever(self.batch_size):
            
            # calculate loss w.r.t to x, y coordinates.
            loss = self.get_loss(x,y)
            
            # calculate weight and bias gradients for two layers.
            weight1_gradient, bias1_gradient, weight2_gradient, bias2_gradient = nn.gradients(loss,[self.weight1, self.bias1, self.weight2, self.bias2])
            
            # based on weight 1 gradient and learning rate update weight 1
            self.weight1.update(weight1_gradient, learning_rate)
            
            # based on bias 1 gradient and learning rate update bias 1
            self.bias1.update(bias1_gradient, learning_rate)
            
            # based on weight 2 gradient and learning rate update weight 2
            self.weight2.update(weight2_gradient, learning_rate)
            
            # based on bias 2 gradient and learning rate update bias 2
            self.bias2.update(bias2_gradient, learning_rate)
            
            # calculate the total loss w.r.t sum of loss of each batch.
            total_loss += nn.as_scalar(loss)*self.batch_size
            
            # sum all the batches to calculate total samples
            total_samples += self.batch_size
            
            # average loss w.r.t total loss.
            average_loss = total_loss/total_samples
            
            # If average loss is less than 0.02, break.
            if(average_loss)<0.02:
            
                break

        return
        
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        # Initialize the batch size and number of features in layers
        self.batch_size = 40
        self.features_layer1 = 300
        self.features_layer2 = 400
        self.features_layer3 = 10
        
        # weight vector of size (number of pixels of gray scale * feature layer1 size)
        self.weight1 = nn.Parameter(784, self.features_layer1)
        
        # weight vector of size (feature layer1 size * feature layer2)
        self.weight2 = nn.Parameter(self.features_layer1, self.features_layer2)
        
        # weight vector of size (feature layer2 size * feature layer3)
        self.weight3 = nn.Parameter(self.features_layer2, self.features_layer3)
        
        # bias vector of size (1, feature layer1 size)
        self.bias1 = nn.Parameter(1, self.features_layer1)
        
        # bias vector of size (1, feature layer2 size)
        self.bias2 = nn.Parameter(1, self.features_layer2)
        
        # bias vector of size (1, feature layer3 size)
        self.bias3 = nn.Parameter(1, self.features_layer3)
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        feature1 = nn.Linear(x, self.weight1)   # create feature layer 1 as linear layer with weight and bias.
        layer1 = nn.AddBias(feature1, self.bias1) 
        layer1 = nn.ReLU(layer1)    # pass layer 1 to ReLU activation function.
        feature2 = nn.Linear(layer1, self.weight2)  # create feature layer 2 as linear layer with weight and bias.
        layer2 = nn.AddBias(feature2, self.bias2)   
        layer2 = nn.ReLU(layer2)                    # pass layer2 through ReLU activation function.
        feature3 = nn.Linear(layer2, self.weight3)
        layer3 = nn.AddBias(feature3, self.bias3)   # create layer 3 with input as layer 2 and bias.
        
        # return layer 3
        return layer3
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # calculate the predicted output by passing input to run function.
        predicted_y = self.run(x)
        
        # calculate loss by passing actual y and predicted y using softmax function.
        loss = nn.SoftmaxLoss(predicted_y, y)
        
        # return loss
        return loss
        
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Initialize 
        learning_rate = -0.05
        num_iter = 1
        total_loss = 0
        total_samples = 0
        
        
        for x, y in dataset.iterate_forever(self.batch_size):
            
            # calculate the loss by passing x and y values.
            loss = self.get_loss(x, y)
            
            # calculate the weight and bias of gradients w.r.t input weights and biases
            weight1_grad, bias1_grad, weight2_grad, bias2_grad, weight3_grad, bias3_grad = nn.gradients(loss,[self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3])
            
            # update weight 1 w.r.t weight 1 gradient and learning rate
            self.weight1.update(weight1_grad, learning_rate)
            
            # update bias 1 w.r.t bias 1 gradient and learning rate
            self.bias1.update(bias1_grad, learning_rate)

            # update weight 2 w.r.t weight 2 gradient and learning rate
            self.weight2.update(weight2_grad, learning_rate)
            
            # update bias 2 w.r.t bias 2 gradient and learning rate
            self.bias2.update(bias2_grad, learning_rate)

            # update weight 3 w.r.t weight 3 gradient and learning rate
            self.weight3.update(weight3_grad, learning_rate)
            
            # update bias 3 w.r.t bias 3 gradient and learning rate
            self.bias3.update(bias3_grad, learning_rate)

            
            if num_iter % 25 == 0:
                
                validationAccuracy = dataset.get_validation_accuracy()
                
                # calculate validation accuracy and if more then 0.98 break.
                if validationAccuracy>=0.975:
                    break
            
            num_iter +=1
        return
        
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        # added number of parameters in each layers
        self.batch_size = 40
        self.l0 = 47
        self.l1 = 256
        self.l2 = 256
        self.l3 = 500
        self.l4 = 5
        
        # Initialized weight and bias parameters.
        self.weight1 = nn.Parameter(self.l0,self.l1)
        self.weight2 = nn.Parameter(self.l1,self.l2)
        self.weight3 = nn.Parameter(self.l2, self.l3)
        self.weight4 = nn.Parameter(self.l3, self.l4)
        self.bias1 = nn.Parameter(1,self.l1)
        self.bias2 = nn.Parameter(1,self.l2)
        self.bias3 = nn.Parameter(1,self.l3)
        self.bias4 = nn.Parameter(1,self.l4)
        
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Initialized the linear parameters
        f_init = nn.Linear(xs[0], self.weight1)
        prev_layer_output = f_init
        
        # For all the examples pass them through linear layer followed by bias.
        for i in range(1, len(xs)):
            f = nn.Add(nn.Linear(xs[i],self.weight1), nn.Linear(prev_layer_output,self.weight2))
            prev_layer_output = f
        f2 = nn.ReLU(nn.AddBias(nn.Linear(prev_layer_output,self.weight3), self.bias3))
        f3 = nn.AddBias(nn.Linear(f2,self.weight4), self.bias4)
        
        # return the output value.
        return f3
        
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # calculate the loss for predicted output.
        predicted_y = self.run(xs)
        loss = nn.SoftmaxLoss(predicted_y, y)
        
        # return the loss calculated by softmax function.
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Initialized the learning rate, number of iterations, total loss and total samples.
        learning_rate = -0.05
        num_iteration = 1
        total_loss = 0
        total_samples = 0
        
        # iterate through the values in dataset, calculate loss, weights and bias for all the...
        #... iterations and break when validation loss is greater than 85%
        for x, y in dataset.iterate_forever(self.batch_size):
            
            # calculating the loss, using loss function
            loss = self.get_loss(x, y)
            
            # calculate the weight and bias parameters from gradients.
            weight1_grad, weight2_grad, weight3_grad, bias3_grad, weight4_grad, bias4_grad = nn.gradients(loss,[self.weight1, self.weight2, self.weight3, self.bias3, self.weight4, self.bias4])
            
            # update weight1 from weight gradient and learning rate using update function.
            self.weight1.update(weight1_grad, learning_rate)
            
            # update weight2 from weight gradient and learning rate using update function.
            self.weight2.update(weight2_grad, learning_rate)
            
            # update weight3 from weight gradient and learning rate using update function.
            self.weight3.update(weight3_grad, learning_rate)
            
            # update bias3 from weight gradient and learning rate using update function.
            self.bias3.update(bias3_grad, learning_rate)

            # update weight4 from weight gradient and learning rate using update function.
            self.weight4.update(weight4_grad, learning_rate)
            
            # update bias4 from weight gradient and learning rate using update function.
            self.bias4.update(bias4_grad, learning_rate)
            
            
            if num_iteration % 25 == 0:
                val_acc = dataset.get_validation_accuracy()
                
                # break the loop if validation accuracy is more than 85%
                if val_acc > 0.85:
                    break
            num_iteration += 1
        return