
/**
 *
 * @author VidhyaLakshmi Venkatarama
 */

package perceptron;

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
import perceptron.*;

public class Main {
    static Perceptron[] perp;
    public static void main(String [] args) {
        
        perp = new Perceptron[13*25];                        // create an array of perceptrons of length Combination(26,2), 1 each for each pair of alphabets.
        int i=0;                                             // index for the array
        for(char alpha ='A'; alpha <='Z'; alpha++){                                   
            for (char alphaN= (char) (alpha+1); alphaN<='Z'; alphaN++){               
                perp[i] = new Perceptron();                                           // create a new instance of perceptron
                perp[i].c1 = alpha;                                                   // the character of the perceptron for which it returns 1
                perp[i].c2 = alphaN;                                                  // the character of the perceptron for which it returns -1
                try {
                    Example[] examples = getExample(alpha, alphaN);                   // call the getExample method to get all the training data for characters alpha and aplhaN
                    perp[i].learn(examples);                                          // run the perceptron learning algorithm
                } catch (IOException ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
                i++;                                                                  // increment the array variable for next perceptron
            }
        }
        testLetter(perp);                                                            //call the method testletter to classify the test data
        int[][] confMat = getConfMat();                                              // get the confusion matrix for the entire test set
        double accur = getAccuracy(confMat);                                         // get the accuracy of the perceptron's learning with the help of confusion matrix
        System.out.println( "accuracy = "+ accur);
        
    }

    /*
    This method reads the files named with the characters. For example if char1 = 'A' and char2 = 'B', this program reads the files A and B which are believed to 
    contain the training data for A and B respectively.
    */
    private static Example[] getExample(char char1, char char2) throws FileNotFoundException, IOException{
        BufferedReader br, br2 = null;
        // 2 lists for training inputs and targets. Lists are used instead of array because the size of the array would vary with the number of input examples which is dynamic
        ArrayList<String[]> values = new ArrayList<>();          //values is a list of array of strings, i.e. the array contains for each example, the target character and the input values
        List<String> chars = new LinkedList<>();                 // chars is a list of the target characters
            int i;
            br = new BufferedReader(new FileReader("test/"+char1));            // br is the buffered reader for character with target =1
            br2 = new BufferedReader(new FileReader("test/"+char2));           // br2 is the buffered reader for character with target = -1
            
            String line1 = null, line2 = null;
            String[] str;
            // read in the lines of training examples for both characters alternatingly
            while (true){
                line1 = br.readLine();                                 // read file (file1) for char1 line by line
                line2 = br2.readLine();                                // read file (file2) for char2 line by line
                if(line1 != null){                                  
                    str = line1.split(",");                           // if line1 is not end of file, split it with respect to comma ,
                    chars.add(str[0]);                                // add the character to the list of target characters
                    values.add((line1.split(",")));                   // add all the comma separated values of the line to the values array
                }
                if (line2 != null){
                    str = line2.split(",");                           // if line12is not end of file, split it with respect to comma ,
                    chars.add(str[0]);                                // add the character to the list of target characters
                    values.add((line2.split(",")));                   // add all the comma separated values of the line to the values array
                }
                if (line1 == null && line2 == null)
                    break;                                           // break if no more lines to read
            }br.close();
            br2.close();
    
            String[][] csvMatrix = values.toArray(new String[values.size()][]); // the matrix csvMatrix is a two dimensional matrix which contains the elements of teh list values
            String[] tar = chars.toArray(new String[chars.size()]);             // the matrix tar contains the elements of the list chars
            Example[] examples1= new Example[tar.length];                       // create an array of type Examples to store all the training examples
            for (i=0; i<examples1.length; i++){
                examples1[i] = new Example();                                   // create a new instance of Example
            }
            
            for (i=0; i<csvMatrix.length; i++){
                
                examples1[i].x[0] =1;                                           // The bias input for a perceptron is always 1
                for (int k=0; k<16; k++){
                    examples1[i].x[k+1] = ((Integer.parseInt(csvMatrix[i][k+1]))*100/15)/100.0;     // scale the input data to be between 0 to 1
                }
                if (tar[i].equals(""+char1)){
                    examples1[i].t=1;                                           // assign the value of target to be 1 if the character is the char1 of perceptron
                }
                else if (tar[i].equals(""+char2)){
                    examples1[i].t= -1;                                         // assign the value of target to be 1 if the character is the char1 of perceptron
                }
                else 
                    System.out.println("You have input the wrong training set");// if the training set contains some target character other than char1 and char2
            }
        return examples1;                                                       // return the array of examples
        
    }
/* This method tests the test data, i.e. runs the perceptron on each set of input data and saves the output against the actual character to the file out.txt.
      
    */
    private static void testLetter(Perceptron[] p) {
        try {
            BufferedReader br = new BufferedReader( new FileReader("test/test_set"));        // file to read the test set
            String line = null;
            BufferedWriter conf = new BufferedWriter(new FileWriter("test/out.txt"));        // file to print the output
            conf.write("Actual   Predicted"+"\n");                    
            while((line = br.readLine()) != null){                                           // while there are input in the test file
                String[] str = line.split(",");                                              // split the line and store into an array
                double[] x = new double[17];                                                 // the array of input per data
                x[0] = 1;
                for (int i=1; i<17; i++){
                    x[i] = (Double.parseDouble(str[i]))/15;                                  // scale the input to be between 0 and 1
                }
                char cF[] = new char[p.length];                                              // character array to store the output of perceptron on the data
                for (int i=0; i<p.length; i++){
                    int y = p[i].runPerceptron(x);                                           // run 325 perceptrons on the data and get the output
                    cF[i] = getCharac(p[i],y);                                               // call the method to return the character corresponding to the output
                }
                char predict = getFinalChar(cF);                                           // get the final prediction taking into account all the 325 perceptrons
                conf.append(str[0]);                                                         // print the actual letter
                conf. append ("           "+predict+ "\n");                                  // print the predicted letter
            }
            conf.close();
            
          
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    // This function returns the character according the perceptron's output
    static char getCharac(Perceptron p,int y){
        if (y==1)
            return p.c1;
        else return p.c2;
    }

    // Return the most popular element in the array cF.
    private static char getFinalChar(char[] cF) {
        char ch;
        int freq[] = new int[26];                                // array to hold the frequency of each character
        int max =0;
        int aVal = Character.getNumericValue('A');
        for (int i=0; i<cF.length; i++){
            int index = Character.getNumericValue(cF[i])-aVal;           // get the index of character cf[i] in the array freq.
            freq[index]++;                                               // increment the frequency of occurence of cf[i] by 1
        }
        for (int i =0; i<26; i++){
            if (freq[i]>freq[max]){
                max =i;                                          // find the index of alphabet with maximum occurence
            }
        }
        ch = (char)('A' + max);                                  // get the alphabet corresponding to that instance
        return ch;                                            
    }

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
            confMatFile = new BufferedWriter(new FileWriter("test/confMat.txt"));             // file to save the confusion matrix in
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
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        return confMat;
    }

    
    // Method to calculate the accuracy as obtained from the confusion matrix
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
    }
}
