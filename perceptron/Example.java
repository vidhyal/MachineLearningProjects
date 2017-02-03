
/**
 *
 * @author VidhyaLakshmi Venkatarama
 */
package perceptron;

public class Example {
    double[] x;                                  // each example contains input vector 'x' and target t
    int t;
    
    Example(){
        this.x = new double[17];                 // each example contains 16 inputs plus one bias input
    }
    
    void printExample(){
        for (int i=0; i<17; i++){
            System.out.print(this.x[i] + " ");
        }
        System.out.println("  "+ this.t);
    }
}
