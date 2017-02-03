
package multineuralnetwork;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Formatter;
import java.util.Random;


/**
 *
 * @author VidhyaLakshmi
 */
public class Network {
    int inputs = 17;       // number of input features + 1 bias input
    int hiddenLayers = 9;  // number of hidden units +bias unit
    int outputs = 26;      // number of output units, 1 per alphabet
    double wHidden[][], wOut[][], dwOut[][], dwHidden[][];
    private double learningRate = 0.3;
    private double alpha=0.3;
    
    Network(){
        wHidden= new double[hiddenLayers][inputs];       // weights from input to hidden units
        wOut = new double[outputs][hiddenLayers];        // weights from hidden units to output
        dwHidden= new double[hiddenLayers][inputs];      // change in weights Hidden between two iterations
        dwOut = new double[outputs][hiddenLayers];       // change in weights Out between two iterations
        
        Random ran = new Random();                      
        for (int j=0; j<hiddenLayers; j++){
            for (int i=0; i<inputs; i++){
                wHidden[j][i]= -0.25+(0.5* ran.nextDouble());  // initialize wHidden with random weights
            }
            for (int k=0; k<outputs; k++){
                wOut[k][j]= -0.25+(0.5* ran.nextDouble());     // initialize wOut with random weights
            }
        }
    }
    
    Result runNetwork(double[] x){
        Result results = new Result(inputs, hiddenLayers, outputs); // create and initialize a new instance of class results
        double output[] = new double[26];
        double h[] = new double[hiddenLayers];
        
        h[0]=1;
        results.hiddenAct[0]= 1;
        for (int j=1; j<hiddenLayers; j++){
            double sum=0;
            for (int i=0;i<inputs; i++){
                sum += wHidden[j][i] *x[i];
            }
            h[j]= sgm(sum);                      // calculate the activation of hidden units
            results.hiddenAct[j]= h[j];          // assign the value to respective field of results 
        }
        
        for (int k=0; k<outputs; k++){
            double sum=0;
            for (int j=0;j<hiddenLayers; j++){
                sum += wOut[k][j] *h[j];
            }
            output[k]= sgm(sum);                // calculate the activation of output units
            results.output[k] = output[k];      // assign the value to respective field of results
        }
        
        System.arraycopy(x, 0, results.input, 0, x.length);  // copyt the value of input activations into results
        return results;
    }
    
    double sgm(double z){
        return (1/( 1 + Math.pow(Math.E,(-z))));          // the definition of sigmoid function
    }
    
    
    // this method is used to learn the network while printing into the accuracy file for getting data for graphs
    void printedLearn(Example[] examples, Example[] test) throws IOException{
       BufferedWriter accFile = new BufferedWriter(new FileWriter("test/results/accuracy.txt"));        // file to print the output
       Formatter fmt = new Formatter(accFile);
       String  str2, str3, str4;
       str2 = "epoch";
       str3 = "Training Accuracy";
       str4 = "Test Accuracy";
       fmt.format("%15s    %15s  %15s\n", str2, str3, str4);
       learn(examples, test, fmt);      // call the learn method
       accFile.close();
    }
/* This method describes the entire learning algorithm of the multineural netwrok. It is private and can not be called by an external class 
        and is called by the method printedLearn to sublearn the network. This method has to be called only after initializing the network. 
        This method takes as input, two arrays of examples of type Example (one for training set and other for test set) and a formatter to 
        print out output. This method calls another method sublearn by passing all the training examples. It then calculates the accuracy of 
        the current network on the training as well as test set while the  current accuracy is less than 1.0 or for a minimum number of epochs,
        and breaks the loop after a maximum number of epochs. If the accuracy goes below the previous accuracy, it copies back the previous 
        values of weight into the weight arrays.*/
    private void learn(Example[] examples, Example[] test, Formatter frmt) {
        int m = examples.length;
        double accuracyTrain =0, prevAcc =-1, accuracyTest=0;
        int epoch =0;
        while (accuracyTrain<1.0 || epoch <200){
           {
            if(epoch>250)
                break;
            
           int accuracyTrainSum=0;                                    // the accuracy of network on the training set
            for (int i=0; i<m; i++){
                char charPred = runNetwork2(examples[i].inputs);      // get the predicted character by running the network (runNetwork2 method) 
                                                                      // on the training set examples
                int character = Character.getNumericValue(charPred)- Character.getNumericValue('A');
                if(character == examples[i].tar) {
                    accuracyTrainSum = accuracyTrainSum+1;           // if the character predicted is same as the actual character, then increment
                                                                    // the accuracysum variable by 1
                }
            }
            prevAcc = accuracyTrain;                                // store the value of accuracy in prevAcc before calculating new accuracy
            accuracyTrain = (double)accuracyTrainSum/m;             // calculate the new accuracyTrain
        }   {
               int accuracyTestSum=0;                               // the accuracy of network on the the test set
               for (int i=0; i<m; i++){
                char charPred = runNetwork2(test[i].inputs);        // get the predicted character by running the network (runNetwork2 method) 
                                                                      // on the test set examples
                int character = Character.getNumericValue(charPred)- Character.getNumericValue('A');
                if(character == test[i].tar) {
                    accuracyTestSum = accuracyTestSum+1;            // if the character predicted is same as the actual character, then increment
                                                                    // the accuracysum variable by 1
                }
            }
            
            accuracyTest = (double)accuracyTestSum/m;                 // calculate the new accuracyTest
            frmt.format("%s", " ");                                   
            frmt.format("%15d    %15f  %15f\n", epoch, accuracyTrain, accuracyTest);  // print the values to the output file
           }
            subLearn(examples);                            // call the method to learn the network
                epoch++;                                   // increase the number of epochs by one
                
        }
        if(prevAcc > accuracyTrain){
               reverseUpdate();                            // if the accuracy on trainingd ata has decreased from the previous one, reset it back by undoing teh update
               accuracyTrain = prevAcc;
           }
        System.out.println(accuracyTrain +"    " + epoch);
        
        
        
    }

    
    private void subLearn(Example[] examples) {
        int m = examples.length;
        Result results[] = new Result[m] ;//= new Result(hiddenLayers, outputs);
        NetError error;
       
        for (int k =0; k<m; k++){
            results[k] = runNetwork(examples[k].inputs);   // call the runNetwork method to run the current network and
                                                           // get the results, i.e. the values of activations for the hidden
                                                           // units and output units
            error = getError(examples[k],results[k]);      // use the results and calculate the error in the network
            runBackPropagation(error, results[k]);         // runbackpropagation with the help of errors to update weights
        }
    }
/* This method takes the results and errors as inputs and runs the backpropagation (of errors) algorithm to update the weights from input 
units to hidden layer units and the weights from hidden units to output units.*/
    private void runBackPropagation(NetError error, Result result) {
        
        for (int k =0; k<outputs; k++){
            for (int j=0; j<hiddenLayers; j++){
                double tempWeight = learningRate * error.dk[k]*result.hiddenAct[j] + alpha* dwOut[k][j];
                                   // the new change in weight contains a factor of gradient and a part of momentum
                dwOut[k][j] = tempWeight;      // update dWOut
                wOut[k][j] = wOut[k][j] + dwOut[k][j];   // update weight.
            }
        }
        for (int j=1; j<hiddenLayers; j++){
            for (int i=0; i<inputs; i++){
                double tempWeight = learningRate* error.dj[j] * result.input[i]+ alpha*dwHidden[j][i];
                                    // the new change in weight contains a factor of gradient and a part of momentum
                dwHidden[j][i] = tempWeight;            // update dWOut
                wHidden[j][i] = wHidden[j][i] + dwHidden[j][i];      // update weight.
            }
        }
    }
/* This method retrieves the error in the network with the help of the training examples' target and the results 
    obtained by running the network on them.
    
    */
    private NetError getError(Example example, Result results) {
        NetError error = new NetError (hiddenLayers);
        double sum=0;
        double target[] = getTarget(example.tar);
        for (int k=0; k<results.output.length; k++){
            double out = results.output[k];
            error.dk[k]= out* (1-out)*(target[k] - out);          //dk = ok * (1-ok)(tk-ok)
            
        }
        for (int j=0; j<hiddenLayers; j++){
            sum =0;
            double hAct = results.hiddenAct[j];
            for (int k=0; k<outputs; k++){
                sum += wOut[k][j]*error.dk[k];
            }
                error.dj[j]=hAct*(1-hAct)*sum;                   // dj = hj* (1-hj)* summation over all outputs(wkj * dk)
        }
        return error;
    
    }

    

/* This method is used while computing the accuracy of the netwrok on the training and test sets. It runs the network on 
    the given example ( the array of doubles) and returns the character predicted by the network.
    */    
    char runNetwork2(double[] x){
        double output[] = new double[26];
        double h[] = new double[hiddenLayers];
        for (int j=1; j<hiddenLayers; j++){
            double sum=0;
            for (int i=0;i<inputs; i++){
                sum += wHidden[j][i] *x[i];          // dot product of weights and inputs calculated iteratively 
            }
            h[j]= sgm(sum);          // calculate the activation of hidden units according to the formula
        }
        h[0]=1;
        for (int k=0; k<outputs; k++){
            double sum=0;
            for (int j=0;j<hiddenLayers; j++){
                sum += wOut[k][j] *h[j];           // dot product of weights and hidden activations calculated iteratively 
            }
            output[k]= sgm(sum);      // calculate the activation of output units according to the formula
        }
        char index = getMax(output);
        return index;
    }
    
    /* This function takes as input the integer corresponding to the target character ( e.g. 0 for 'A', 25 for 'Z')
and returns an array of doubles with 0.9 for the index corresponding to the target.( e.g. if target = 'A', then target[0] =0.9 
    and for all other integer i from 0 to 25, target[i]= 0.1.)
*/
private double[] getTarget(int tar) {
        double target[] = new double[26];
        for (int k =0; k<26; k++){
                target[k]= 0.1;
        }
        target[tar] = 0.9;
        return target;
    }
/*
this method reads the output activations and gets the maximum value out of it and returns the character corresponding 
to the output unit with maximum activation.
*/
    private char getMax(double[] output) {
        char ch;
        int max =0;
        for (int i =0; i<output.length; i++){
            if (output[i]> output[max]){
                max = i;
            }
        }
        ch = (char)('A' + max);                                  // get the alphabet corresponding to that instance
        return ch;      

    }

    /* this method simply undos an update of weights when the previous accuracy obtained was larger than the 
    current accuracy
    */
    private void reverseUpdate() {
        for (int i=0; i<inputs; i++){
            for(int j = 0; j<hiddenLayers; j++)
                wHidden[j][i] = wHidden[j][i]- dwHidden[j][i];
        }
        for (int k=0; k<outputs; k++){
            for(int j = 0; j<hiddenLayers; j++)
                wOut[k][j] = wOut[k][j]- dwOut[k][j];
        }
    }
   
}
