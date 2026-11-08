
package numj;

import numj.core.NumJArray;

public class NeuralNet {

    public static final int INPUTS = 794;
    public static final int HIDDEN1 = 128;
    public static final int HIDDEN2 = 64;
    public static final int OUTPUTS = 10;

    public static final float LEARNING_RATE = 0.001f;

    // Weights and biases
    public NumJArray weights1;
    public NumJArray bias1;
    public NumJArray weights2;
    public NumJArray bias2;
    public NumJArray weights3;
    public NumJArray bias3;

    public NumJArray test = new NumJArray(new float[]{0, 0, 1}, new int[]{3});
    public NumJArray test2 = new NumJArray(new float[]{0, 0, 1000}, new int[]{3});
    public NumJArray test3 = new NumJArray(new float[]{0, 0, 0}, new int[]{3});

    public NumJArray y = new NumJArray(new float[]{1, 0, 0, 0, 1, 0}, new int[]{3, 2});
    public NumJArray logit = new NumJArray(new float[]{2, 1, 0, 0, 3, 0}, new int[]{3, 2});


    // Network data
    public NumJArray input;
    public NumJArray correct;

    // Forward pass values
    public NumJArray z1, a1;
    public NumJArray z2, a2;
    public NumJArray logits, output;

    // Gradients
    public NumJArray dW1, db1;
    public NumJArray dW2, db2;
    public NumJArray dW3, db3;

    public float loss;

    public void createArrays() {

        weights1 = NumJArray.uniform(-0.1f, 0.1f, INPUTS, HIDDEN1);
        bias1 = NumJArray.zeros(HIDDEN1);

        weights2 = NumJArray.uniform(-0.1f, 0.1f, HIDDEN1, HIDDEN2);
        bias2 = NumJArray.zeros(HIDDEN2);

        weights3 = NumJArray.uniform(-0.1f, 0.1f, HIDDEN2, OUTPUTS);
        bias3 = NumJArray.zeros(OUTPUTS);

        input = NumJArray.ones(INPUTS);

        correct = new NumJArray(
            new float[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            new int[]{OUTPUTS}
        );
    }

    // FORWARD PASS
    public void predict() {

        // Layer 1
        z1 = input.matMul(weights1).add(bias1);
        a1 = z1.relu();

        // Layer 2
        z2 = a1.matMul(weights2).add(bias2);
        a2 = z2.relu();

        // Output layer
        logits = a2.matMul(weights3).add(bias3);
        output = logits.softmax();
    }

    // LOSS
    public void getError() {

        // Reshape to [1, OUTPUTS] because smce uses axis 1
        loss = correct.reshape(1, OUTPUTS).smce(logits.reshape(1, OUTPUTS));
    }

    // BACKPROPAGATION
    public void backPropagate() {

        // Output layer gradient
        NumJArray outputError = output.sub(correct);

        dW3 = a2.outer(outputError);
        db3 = outputError;

        // Hidden layer 2
        NumJArray hidden2Error = outputError.matMul(weights3.transpose()).mul(z2.reluD());

        dW2 = a1.outer(hidden2Error);
        db2 = hidden2Error;

        // Hidden layer 1
        NumJArray hidden1Error = hidden2Error.matMul(weights2.transpose()).mul(z1.reluD());

        dW1 = input.outer(hidden1Error);
        db1 = hidden1Error;
    }

    // UPDATE WEIGHTS
    public void updateWeights() {

        weights3 = weights3.sub(dW3.mul(LEARNING_RATE));
        bias3 = bias3.sub(db3.mul(LEARNING_RATE));

        weights2 = weights2.sub(dW2.mul(LEARNING_RATE));
        bias2 = bias2.sub(db2.mul(LEARNING_RATE));

        weights1 = weights1.sub(dW1.mul(LEARNING_RATE));
        bias1 = bias1.sub(db1.mul(LEARNING_RATE));
    }

    public static void main(String[] args) {

        NeuralNet nn = new NeuralNet();
        nn.createArrays();

        nn.predict();
        nn.getError();

        System.out.println("Initial loss: " + nn.loss);

        for (int i = 0; i < 1000; i++) {

            nn.predict();
            nn.getError();

            nn.backPropagate();
            nn.updateWeights();
        }

        // Calculate loss again using updated weights
        nn.predict();
        nn.getError();

    }
}
