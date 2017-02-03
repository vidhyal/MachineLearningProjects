/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multineuralnetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author VidhyaLakshmi
 */
public class MultiNeuralNetwork {

    static double avg[] = new double[16];
    static double stdDev[] = new double[16];
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)  {
        
            Network network = new Network();
         try {   
            Example[] examples = getExample();                   // call the getExample method to get all the training data 
            Example[] examplesTest = getTestData();              //get all the test data
            network.printedLearn(examples, examplesTest);       // run the network learning algorithm
       
        } catch (IOException ex) {
            Logger.getLogger(MultiNeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        }
        testLetter(network);                                      // test the test data on the network
        int[][] confMat = getConfMat();                           // get the confusion matrix for the entire test set
        double accur = getAccuracy(confMat);                      // get the accuracy of the network on test data with the help of confusion matrix
        System.out.println( "accuracy = "+ accur);
        
    }
    static Example[] getExample() throws FileNotFoundException, IOException{
        
        
        ArrayList<String[]> values = new ArrayList<>();          //values is a list of array of strings, i.e. the array contains for each example, the target character and the input values
        List<String> chars = new LinkedList<>();                 // chars is a list of the target characters
            int i;
            int aVal = Character.getNumericValue('A');
            BufferedReader br = new BufferedReader(new FileReader("test/training_set"));            // br is the buffered reader for character with target =1
            
             String line1 = null;
            String[] str;
            while ((line1 = br.readLine()) != null){              // read the training data line by line
           
                str = line1.split(",");                           // if line1 is not end of file, split it with respect to comma ,
                chars.add(str[0]);                                // add the character to the list of target characters
                values.add((line1.split(",")));                   // add all the comma separated values of the line to the values array
            }
            br.close();
            
    
            String[][] csvMatrix = values.toArray(new String[values.size()][]); // the matrix csvMatrix is a two dimensional matrix which contains the elements of teh list values
            String[] tar = chars.toArray(new String[chars.size()]);             // the matrix tar contains the elements of the list chars
            Example[] examples1= new Example[tar.length];                       // create an array of type Examples to store all the training examples
            for (i=0; i<examples1.length; i++){
                examples1[i] = new Example();                                   // create a new instance of Example
            }
            
            double xOrig[][] = new double[csvMatrix.length][16];
            for (i=0; i<csvMatrix.length; i++){
                for (int k=0; k<16; k++){
                    xOrig[i][k] = ((Integer.parseInt(csvMatrix[i][k+1])));
                }
                examples1[i].tar = Character.getNumericValue(tar[i].charAt(0))- Character.getNumericValue('A'); 
            }
                avg = average(xOrig);                                           // Get the average of every feature over the input data
                stdDev = standardDeviation(xOrig, avg);                         // Get the standard deviation of every feature over the input data
            for (i=0; i<csvMatrix.length; i++){
                examples1[i].inputs[0] =1;                                      // The bias input is always 1
                for (int k=0; k<16; k++){
                    examples1[i].inputs[k+1] = standardize(xOrig[i][k], k);     // preprocess the input data to belong to a standard distribution
                }
            }
        return examples1;                                                       // return the array of examples
    }

    private static double[] average(double[][] xOrig) {
        for (int i=0; i<xOrig[0].length; i++){
            int sum =0;
            for (double[] xOrig1 : xOrig) {
                sum += xOrig1[i];
            }
            avg[i]= (double)sum/xOrig.length;
        }
        return avg;
        
    }

    private static double[] standardDeviation(double[][] xOrig, double avg[]) {
        //double stdDev[] = new double[xOrig[0].length];
        for (int i=0; i<xOrig[0].length; i++){
            double sum =0;
            for (double[] xOrig1 : xOrig) {
                sum += Math.pow((xOrig1[i] - avg[i]),2);
            }
            stdDev[i]= Math.pow((double)sum/xOrig.length, 0.5);
        }
        return stdDev;
    }

    private static void testLetter(Network network) {
         try {
            BufferedReader br = new BufferedReader( new FileReader("test/test_set"));        // file to read the test set
            String line = null;
            BufferedWriter conf = new BufferedWriter(new FileWriter("test/out.txt"));        // file to print the output
            conf.write("Actual   Predicted"+"\n");                    
            while((line = br.readLine()) != null){                                           // while there are input in the test file
                String[] str = line.split(",");                                              // split the line and store into an array
                double[] x = new double[17];                                                 // the array of input per data
                x[0] = 1;
                for (int i=0; i<16; i++){
                    x[i+1] = standardize ((Double.parseDouble(str[i+1])),i);                // standardize the input with respect to the standardization
                                                                                            //parameters of training data
                }
                char predict = network.runNetwork2(x) ;                                     // get the predicted character
                conf.append(str[0]);                                                         // print the actual letter
                conf. append ("           "+predict+ "\n");                                  // print the predicted letter
            }
            conf.close();
            
          
        } catch (IOException ex) {
            Logger.getLogger(MultiNeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.

    private static int[][] getConfMat() {
        int[][] confMat= new int[26][26];                                          // a two dimensional array to store the confusion matrix
                                                                                   // rows correspond to actual character and column to predicted character
        int i,j;           
        BufferedWriter confMatFile = null;                                         // file to store the confusion matrix
        try {
            BufferedReader conf = new BufferedReader(new FileReader("test/out.txt"));  // file reader for out.txt
            String line=null;
            conf.readLine();                                                          // ignore the title line
            while((line = conf.readLine())!=null){                                    // while there is data to read in out.txt
                String[] str = line.split("\\s+");                                     // split the characters based on whitespaces
                i = Character.getNumericValue(str[0].charAt(0)) -Character.getNumericValue('A');       // get the index of actual character
                j = Character.getNumericValue(str[1].charAt(0)) -Character.getNumericValue('A');       // get the index of predicted character
                confMat[i][j]= confMat[i][j]+1;                                                     // increment the value of the cell corresponding to actual character with index i and predicted character with index j
            }
            confMatFile = new BufferedWriter(new FileWriter("test/results/confMat.txt"));             // file to save the confusion matrix in
            Formatter fmt = new Formatter(confMatFile);                                      // to print a formatted array                        
            System.out.print(" ");
            fmt.format("%s", " ");                                    
            // print out the entire confusion matrix
            for (i=0; i<26; i++){
                System.out.printf("%5s", (char)('A'+i));
                fmt.format("%5s", (char)('A'+i));
            }
            for (i=0; i<26; i++){
                System.out.print("\n" +(char)('A'+i));
                fmt.format("\n" +(char)('A'+i));
                for (j=0; j<26; j++){
                    System.out.printf("%5d",confMat[i][j]);
                    fmt.format("%5d",confMat[i][j]);
                }
                System.out.println();
            }
                confMatFile.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MultiNeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MultiNeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
        }
        return confMat;
    
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private static double getAccuracy(int[][] confMat) {
         int sum =0;
        for (int i=0; i<26; i++){
            for (int j=0; j<26; j++){
                sum = sum+confMat[i][j];
            }
        }
        int accurate= 0;
        for (int i=0; i<26; i++){
            accurate = accurate+ confMat[i][i];
        }
        return ((double)accurate/sum);
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private static double standardize(double d, int k) {
        double input;
                    if (!(stdDev[k]==0))
                        input = ((double)d - avg[k])/stdDev[k];
                    else
                      input =((double)d - avg[k]);// d;
                    
                
        return input;
         
    }

    private static Example[] getTestData() throws FileNotFoundException, IOException {
        ArrayList<String[]> values = new ArrayList<>();          //values is a list of array of strings, i.e. the array contains for each example, the target character and the input values
        List<String> chars = new LinkedList<>();                 // chars is a list of the target characters
            int i;
            int aVal = Character.getNumericValue('A');
            BufferedReader br = new BufferedReader(new FileReader("test/test_set"));            // br is the buffered reader for character with target =1
            
             String line1 = null;
            String[] str;
            int lines =0;
            // read in the lines of training examples for both characters alternatingly
            while ((line1 = br.readLine()) != null){                                 // read file (file1) line by line
           //line1 = br.readLine();
           lines++;
           //if (lines>)
             //  break;
                str = line1.split(",");                           // if line1 is not end of file, split it with respect to comma ,
                chars.add(str[0]);                                // add the character to the list of target characters
                values.add((line1.split(",")));                   // add all the comma separated values of the line to the values array
            }
            br.close();
            
    
            String[][] csvMatrix = values.toArray(new String[values.size()][]); // the matrix csvMatrix is a two dimensional matrix which contains the elements of teh list values
            String[] tar = chars.toArray(new String[chars.size()]);             // the matrix tar contains the elements of the list chars
            Example[] examples1= new Example[tar.length];                       // create an array of type Examples to store all the training examples
            for (i=0; i<examples1.length; i++){
                examples1[i] = new Example();                                   // create a new instance of Example
            }
            //double avg[] = new double[16];
            //double stdDev[] = new double[16];
            double xOrig[][] = new double[csvMatrix.length][16];
            for (i=0; i<csvMatrix.length; i++){
                
               
                for (int k=0; k<16; k++){
                    xOrig[i][k] = ((Integer.parseInt(csvMatrix[i][k+1])));
                    
                }
               examples1[i].tar = Character.getNumericValue(tar[i].charAt(0))- Character.getNumericValue('A'); 
            }
            for (i=0; i<csvMatrix.length; i++){
                examples1[i].inputs[0] =1;                                           // The bias input is always 1
                for (int k=0; k<16; k++){
                    examples1[i].inputs[k+1] = standardize(xOrig[i][k], k);
           
                    
                }
           
            }
        return examples1;                                                       // return the array of examples
        
        
    }
    

    
    
}
