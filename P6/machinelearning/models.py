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
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dot_product = self.run(x)
        score = nn.as_scalar(dot_product)
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False

        while not converged:
            converged = True 

            for x, y in dataset.iterate_once(1): 
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):  
                    self.w.update(x, nn.as_scalar(y))
                    converged = False  

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.hidden_size = 512
        self.learning_rate = 0.05

        self.weights1 = nn.Parameter(1, self.hidden_size)
        self.bias1 = nn.Parameter(1, self.hidden_size)

        self.weights2 = nn.Parameter(self.hidden_size,1)
        self.bias2 = nn.Parameter(1, 1)
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        hidden_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.weights1), self.bias1))

        output_layer = nn.AddBias(nn.Linear(hidden_layer, self.weights2), self.bias2)

        return output_layer


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        loss = nn.SquareLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(200):
            loss = self.get_loss(x, y)
            gradients = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2])

            self.weights1.update(gradients[0], -self.learning_rate)
            self.bias1.update(gradients[1], -self.learning_rate)
            self.weights2.update(gradients[2], -self.learning_rate)
            self.bias2.update(gradients[3], -self.learning_rate)

            if(nn.as_scalar(loss)<0.018):
                break

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
        self.hidden_size = 200
        self.learning_rate = 0.5

        self.weights1 = nn.Parameter(784, self.hidden_size)
        self.bias1 = nn.Parameter(1, self.hidden_size)

        self.weights2 = nn.Parameter(self.hidden_size, 10)
        self.bias2 = nn.Parameter(1, 10)

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
        hidden_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.weights1), self.bias1))

        output_layer = nn.AddBias(nn.Linear(hidden_layer, self.weights2), self.bias2)

        return output_layer

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
        predictions = self.run(x)
        loss = nn.SoftmaxLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        max_epochs = 20  
        target_accuracy = 0.975  
        count=0
        for epoch in range(max_epochs):
            for x, y in dataset.iterate_forever(100):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2])
                count+=1
                
                self.weights1.update(gradients[0], -self.learning_rate)
                self.bias1.update(gradients[1], -self.learning_rate)
                self.weights2.update(gradients[2], -self.learning_rate)
                self.bias2.update(gradients[3], -self.learning_rate)
                if(count==600):
                    count=0
                    break
                    
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= target_accuracy:
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
        self.hidden_size = 64
        self.learning_rate = 0.01

        self.weights_hh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.weights_xh = nn.Parameter(47, self.hidden_size)  
        self.bias_h = nn.Parameter(1, self.hidden_size)

        self.weights_out = nn.Parameter(self.hidden_size, 5)  
        self.bias_out = nn.Parameter(1, 5)

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
        h_t = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.weights_xh), self.bias_h))

        for x_t in xs[1:]:
            h_t = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(h_t, self.weights_hh), nn.Linear(x_t, self.weights_xh)), self.bias_h))

        output_layer = nn.AddBias(nn.Linear(h_t, self.weights_out), self.bias_out)

        return output_layer

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
        predictions = self.run(xs)
        loss = nn.SoftmaxLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        max_epochs = 20
        target_accuracy = 0.85
        batch_size = 32
        count=0
        for epoch in range(max_epochs):
            for xs, y in dataset.iterate_forever(batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.weights_hh, self.weights_xh, self.bias_h, self.weights_out, self.bias_out])

                self.weights_hh.update(gradients[0], -self.learning_rate)
                self.weights_xh.update(gradients[1], -self.learning_rate)
                self.bias_h.update(gradients[2], -self.learning_rate)
                self.weights_out.update(gradients[3], -self.learning_rate)
                self.bias_out.update(gradients[4], -self.learning_rate)
                count+=1
                if(count==1000):
                    count=0
                    break  
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= target_accuracy:
                break

