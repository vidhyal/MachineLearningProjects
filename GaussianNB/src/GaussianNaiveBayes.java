package gaussiannaivebayes;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.log;
import java.util.Formatter;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author VidhyaLakshmi
 */
public class GaussianNaiveBayes {

    private static double[] getProb() throws FileNotFoundException, IOException {
        BufferedReader probFile = new BufferedReader(new FileReader("test/prob.txt"));
        int total = Integer.parseInt(probFile.readLine().split("\\s+")[1]);
        int pos = Integer.parseInt(probFile.readLine().split("\\s+")[1]);
        int neg = Integer.parseInt(probFile.readLine().split("\\s+")[1]);
        double prob[] = new double[2];
        prob[0] = (double)(pos)/total;
        prob[1]=(double)(neg)/total;
        return prob;
        
    }

    /**
     * @param args the command line arguments
     */
   static  HashMap<String,double[]> params = new HashMap<>();
    public static void main(String[] args) throws FileNotFoundException, IOException {
        BufferedReader paramsFile = new BufferedReader(new FileReader("test/Gaussian_parameters.txt"));
       // read the file containing gausian parameters and store the arrays in hashmap
       double fix = 0.000000001;
        String line;
        while((line= paramsFile.readLine())!=null){
            String str=line.split("\n")[0];
            double[] paramArray = new double[57];
            line=paramsFile.readLine();
            String param[] =line.split("\\s+");
            Boolean std =false;
            if((str.split("_")[0]).equals("stdDev")){
                std=true;
            }
            for(int i=0; i<param.length-1;i++){
                if(std && Double.parseDouble(param[i+1])== 0)
                    paramArray[i] = fix;                   // add a small quantity to zero standarad deviation to avoid division by zero
                else
                    paramArray[i] = Double.parseDouble(param[i+1]);
            }
            params.put(str, paramArray);
        }
        testData(params); // call the method to classify test data
        
    }

    private static void testData(HashMap<String, double[]> params) throws FileNotFoundException, IOException {
        ReadTestFeatures();       
        testGaussian();      // test using the Gaussian Naive Bayes 
        int confMat[][] =getConfMat("test/outTemp.txt");
        printConfMat(confMat);
    }
    
    private static void ReadTestFeatures() {
        BufferedReader reader;
        try{
            reader=new BufferedReader(new FileReader("test/test_features.data")); //file containg test data
            String line;
            int lineNum=0;                                                            // maintain an index for nth test input
            while((line=reader.readLine())!=null ){                                   
               // featuresTesting.put(lineNum, new double[57]);
                String[] tokens=line.split(" ");                                      // get the value of label and featureIndex-value  pair
                int label=(int)Double.parseDouble(tokens[0]);                         // label is the first integer in the line
                labelTesting.put(lineNum, label);                                     // add the label for lineNum'th example into hashmap
                double[] input = new double[57];
                for(int i=1;i<tokens.length;i++){
                    String[] fields =tokens[i].split(":");                            //ignore first token (label) and get featureIndex-value pairs  
                    input[i-1]=Double.parseDouble(fields[1]);                           //get the value of feature
                }
                featuresTesting.put(lineNum,input);        // add the pair to Hashmap
            lineNum++;
            }
            reader.close();
        }catch (Exception e){
            System.out.println(" "+e.getMessage());
        }
    }
    
    static HashMap<Integer, double[]> featuresTesting=new HashMap<>();
    static HashMap<Integer, Integer> labelTesting=new HashMap<>();
    static double finAccuracy;
    private static void testGaussian() throws IOException {
        int accuracySum=0;
        
        BufferedWriter out = new BufferedWriter (new FileWriter("test/outTemp.txt"));
        out.write("Actual Predicted  \n");
        double prob[] = getProb();
        Random ran = new Random();
        for (Integer testInstance:featuresTesting.keySet()){
            double[] temp = featuresTesting.get(testInstance);
            // get the log of a posteriori of each class according to MAP estimate
            double prob_pos = getValue("pos",temp,prob[0]);
            double prob_neg = getValue("neg",temp,prob[1]);
            int d = -1;
            if(prob_pos>prob_neg)
                d = 1;
            else if (prob_pos == prob_neg)
                d = ran.nextInt(1);          //randomly decide the class when both classes show equal probability
            out.append(labelTesting.get(testInstance) + "     " +d +"\n");
         
            if (d==labelTesting.get(testInstance))
                accuracySum++;
        }
        finAccuracy = (double)accuracySum/labelTesting.size();
        System.out.println("accuracy= " +finAccuracy);
        out.close();
       
    }

    private static double getValue(String clas, double[] temp, double probInit) {
        double mean[],stdDev[],prob;
        if (clas.equals("pos")){
            mean = params.get("means_pos");
            stdDev = params.get("stdDev_pos");
        }
        else{
            mean = params.get("means_neg");
            stdDev = params.get("stdDev_neg");
        }
        prob = log(probInit); //log of prior probability of class
        for (int i=0; i<temp.length; i++){
            prob += log(gaussian(temp[i],mean[i],stdDev[i]));  // for each features, add the log of their conditional probability
        }
        return prob;
    }

    private static double gaussian(double x, double mean, double stdDev) {
        // return the Gaussian distribution value for given value of feature, mean and standard deviation
        return (1/(Math.pow((2*Math.PI),0.5)*stdDev)* Math.exp(-1*(Math.pow(((x-mean)/stdDev),2)/2)));
    }
    
        private static int[][] getConfMat(String testoutTemptxt) {
        int confMat[][] = new int [2][2];
        int i,j;           
        try {
            BufferedReader outFile = new BufferedReader(new FileReader(testoutTemptxt));  // file reader for out.txt
            String line=null;
            outFile.readLine();                                                          // ignore the title line
            while((line = outFile.readLine())!=null){                                    // while there is data to read in out.txt
                String[] str = line.split("\\s+");                                     // split the characters based on whitespaces
                i = Integer.parseInt(str[0]) ;
                if(Integer.parseInt(str[0])==1)
                     i=0;
                else i=1;   // i is actual, j is predicted
                 if(Double.parseDouble(str[1])==1)
                     j =0;// Integer.parseInt(str[1]);
                 else j=1;
                confMat[i][j]= confMat[i][j]+1;                                                     // increment the value of the cell corresponding to actual character with index k and predicted character with index j
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(GaussianNaiveBayes.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(GaussianNaiveBayes.class.getName()).log(Level.SEVERE, null, ex);
        }
        return confMat;
    }
         private static void printConfMat(int[][] confMat) throws IOException {
             BufferedWriter conf = new BufferedWriter(new FileWriter("test/confMat.txt"));
        Formatter fmt = new Formatter(conf);                                      // to print a formatted array                        
        fmt.format("%4s", " ");  
        fmt.format("%6s%6s\n", "pos", "neg");
        for (int i=0; i<2; i++){
            String str;
            if (i==0) 
                str="pos";
            else str="neg";
            fmt.format("%4s",str);
            for (int j=0; j<2; j++){
                fmt.format("%6d",confMat[i][j]);
            }
            fmt.format("\n");
        }
        int tp, fp, fn;
        tp = confMat[0][0];
        fp = confMat[1][0];
        fn = confMat[0][1];
        double precision = (double)tp/(tp +fp);
        double recall = (double)tp/(tp+fn);
       conf.append("accuracy = " +finAccuracy+"\nprecision = "+ precision+ "\nrecall = "+recall);
       conf.close();
        System.out.println ("precision"+precision+"   recall"+ recall);
    }   
}
