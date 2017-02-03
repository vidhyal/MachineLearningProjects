
/**
 *
 * @author VidhyaLakshmi Venkatarama
 */
package perceptron;

import java.util.Random;

public class Perceptron {

    int numberofInputs = 16;
    double w[], prevW[];
    
    double learningRate = 0.2;
    double accuracy=0, prevAcc = -1;
    char c1, c2;
	
	Perceptron() {
		w = new double[numberofInputs +1];              // weights w[1] to w[16] corresponding to each input and w[0] is bais
                prevW = new double[numberofInputs +1];         // an array for weights calculated in the previous learning iteration
		Random ran = new Random();                    
                for (int i=0; i<numberofInputs; i++){
			w[i]= (-1.0 + (2 * ran.nextDouble()));    // randomly assign weights in the range -1 to 1
		}
	}
        
        /* This method describes the entire learning algorithm of the perceptron. It is public and called by an external class 
        to sublearn a perceptron. This method has to be called only after initializing the perceptron.
        This method takes as input, an array of examples of type Example (also known as training set) and calls another method 
        sublearn by passing all the examples. It then calculates the accuracy of the current perceptron on that set while the 
        current accuracy is larger than the previous accuracy, and breaks teh loop after a maximum number of epochs. If the 
        accuracy goes below the previous accuracy, it copies back the previous values of weight into the weight arrays.
        There are two more variations of the method where, in addition to the previous conditions, the while loop runs for 
        atleast a ceratin number of epochs or until a particular accuracy is achieved. This is done to get out of local minima.
        .*/
        void learn(Example[] examples){
            int m = examples.length;                         // number of training examples
            int epoch =0;                                    // corresponds to 1 epoch of the learning algorithm
            int accuracySum;
            while ((prevAcc< accuracy-0.00001)||epoch <25){// || accuracy<0.9){
                {
                    System.arraycopy(w, 0, prevW, 0, 17);             // copy new values of weights to the array with old values.
                    sublearn(examples);                               // call the sublearn method to obtain new set of weights
                    prevAcc = accuracy;                               // store the value of accuracy in prevAcc before calculating new accuracy      
                    accuracySum =0;
                    for (int i=0; i<m; i++){
                        if (examples[i].t == runPerceptron(examples[i].x)){
                            accuracySum = accuracySum+1;
                        }
                    }
                    accuracy = (double)accuracySum/m;                 // calculate the new accuracy
                }
                epoch++;                                             // increment epoch
                //if (epoch>=65 )                                      // if epoch is greter than the maximum number of allowed epochs, break.
                  //  break;                                           // this is to avoid overfitting
            }
            if (accuracy < prevAcc){
                System.arraycopy(prevW, 0, w, 0, 17);               // use the weights that gave a higher accuracy.
                accuracy = prevAcc;                                 
            }
            
        }
	
        /* This method is private and is called by the learn method. It runs the perceptron with the present values of weights for each example
        and if the output of the perceptron is different from the target value, it increments the weights by the product of learning rate, the 
        corresponding input and the target.
        */
	private void sublearn(Example[] examples){
            int m = examples.length;
		for (int k =1; k<m; k++ ){
			int y = runPerceptron (examples[k].x);
			if (y != examples[k].t ){
                            for( int i=0; i<=numberofInputs; i++){
                                double dw =learningRate * examples[k].x[i] *examples[k].t;   // increase the weight proportional to the gradient.
                                w[i] = w[i]+dw;
                            }
                        }
		}
	}
        /* This method returns the output of perceptron. i.e., it returns 1 if the dot product of weight vector and input vector is greater than 
        or equal threshold, else -1. */
        int runPerceptron(double []x){
            double sum =0;
            for( int i=0; i<17; i++){
                sum = sum+ w[i]*x[i];          // final sum =w.x+b
            }
            if (sum<0)
                return -1;                    // if sum is less than zero,i.e, perpceptron's output is less than threshold, y = -1
            else
            return 1;                        // else, y=1
        }
}
