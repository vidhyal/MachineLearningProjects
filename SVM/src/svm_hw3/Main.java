/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package svm_hw3;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Formatter;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import libsvm.*;
import static java.lang.Math.abs;
/**
 *
 * @author VidhyaLakshmi
 */


        
public class Main {
static HashMap<Integer, HashMap<Integer, Double>> featuresTesting=new HashMap<>();
static HashMap<Integer, Integer> labelTesting=new HashMap<>();
static HashMap<Integer, HashMap<Integer, Double>> featuresTraining=new HashMap<>();
static HashMap<Integer, Integer> labelTraining=new HashMap<>();
static int lengthTillNow=0;
static int kFold =10;
BufferedWriter conf ;
static double cParam;


    public static void main(String[] args) throws IOException {
        ReadTrainingFeatures();
        ReadTestFeatures();
        
        svm_model model = runExp1();  // run experiment 1 and return the final model
        runExp2(model);               // use the model from experiment 1 to run experiment 2
        runExp3();                    // run experiment 3
    }
    // This function reads the training data file and loads the data into two Hashmaps featuresTraining and labelTraining
    private static void ReadTrainingFeatures() {
         BufferedReader reader;
        try{
            reader=new BufferedReader(new FileReader("test/shuff_train2.data"));   // read from the file which contains shuffled training data
            String line;
            int lineNum=0;                                                         // maintain an index for nth training example
            while((line=reader.readLine())!=null ){                                //read the file line by line
                featuresTraining.put(lineNum, new HashMap<>());
                String[] tokens=line.split(" ");                                   // get the value of label and featureIndex-value  pair
                int label=(int)Double.parseDouble(tokens[0]);                      // label is the first integer in the line
                labelTraining.put(lineNum, label);                                 // add the label for lineNum'th example into hashmap
                for(int i=1;i<tokens.length;i++){                                  
                    String[] fields =tokens[i].split(":");                         //ignore first token (label) and get featureIndex-value pairs
                    int featureId=Integer.parseInt(fields[0]);                     // parse the featureIndex
                    double featureValue=Double.parseDouble(fields[1]);             //get the value of feature
                    featuresTraining.get(lineNum).put(featureId, featureValue);    // add the pair to Hashmap
                }
            lineNum++;
            }
            reader.close();
        }catch (Exception e){
            System.out.println(" "+e.getMessage());
        }
    }
// This function reads the test data file and loads the data into two Hashmaps featuresTesting and labelTesting
    private static void ReadTestFeatures() {
        BufferedReader reader;
        try{
            reader=new BufferedReader(new FileReader("test/svm_test_features.data")); //file containg test data
            String line;
            int lineNum=0;                                                            // maintain an index for nth test input
            while((line=reader.readLine())!=null ){                                   
                featuresTesting.put(lineNum, new HashMap<>());
                String[] tokens=line.split(" ");                                      // get the value of label and featureIndex-value  pair
                int label=(int)Double.parseDouble(tokens[0]);                         // label is the first integer in the line
                labelTesting.put(lineNum, label);                                     // add the label for lineNum'th example into hashmap
                for(int i=1;i<tokens.length;i++){
                    String[] fields =tokens[i].split(":");                            //ignore first token (label) and get featureIndex-value pairs  
                    int featureId=Integer.parseInt(fields[0]);                        // parse the featureIndex
                    double featureValue=Double.parseDouble(fields[1]);                //get the value of feature
                    featuresTesting.get(lineNum).put(featureId, featureValue);        // add the pair to Hashmap
                }
            lineNum++;
            }
            reader.close();
        }catch (Exception e){
            System.out.println(" "+e.getMessage());
        }
    }
    /*This method calls for a k-fold cross validation, teh value of k-dicteted by the global variable kFold and gets a parameter c for the SVM.
    The method further issues calls for creating a final model and testing the SVM model on the test data. It further calls a method to obtain data 
    needed to construct teh ROC curve.
    */
    private static svm_model runExp1() throws IOException {
        svm_problem mainProb = getMainProblem();                                 // get SVM problem, i.e the training set in svm format
                
        svm_parameter param = new svm_parameter();                                // set all the parameters needed for the linear binary SVM classifier
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.LINEAR;
        
        cParam = runkFoldValidation(param,mainProb.l);                           // run the kFoldValidation and obtain best c and update the glbal variable cParam
        param.C = cParam;
        svm_model finalmodel =  svm.svm_train(mainProb, param);                  // train the SVM on the training data (mainProb)

        testSVM(finalmodel);                                                     //test SVM on test data
        runThresholdCurve();                                                     //call the method to obtain precision and recall
        return finalmodel;                                                       // return the final model needed for experiment 2
   }
/* This method calculates the weight vector as indicated by the model passed , into an array. It then sorts the array of weights in ascending order
    and gets the corresponding indices. It then uses these indices to get the 'm' features with highest weights for all integer m in 2 to 57. On 
    obtaining these features, the method tehn sends for training and testing SVM based on these m features.
*/
    private static void runExp2(svm_model model) throws IOException {
        double weights[] = getWeights(model);                                       // obtain teh array of weights (weight vector)
        Integer[] sortedIndex = sortWeights(weights);                               // sort the array of weights and get the sorted indices
        for (int i=0; i<sortedIndex.length;i++)
            sortedIndex[i] +=1;                                                     // since the index in java starts from 0 and our features start from 1,
                                                                                    // add one to all the indices
    try (BufferedWriter out = new BufferedWriter (new FileWriter("test/outTempExp2.txt"))) { // file to store the m vs. accuracy data
        out.write("m      Accuracy \n");
        for (int m = 2; m<=57; m++){
            System.out.println(m);
            int featIndex[] = getMlargestIndices(sortedIndex, m);                   // get the top m features
            runSVM(featIndex, out);                                                 // run SVM (train and test) on these m features
        }
    }
    }

    private static void runExp3() throws IOException {
        try (BufferedWriter out = new BufferedWriter (new FileWriter("test/outTempExp3.txt"))) {
            out.write("m      Accuracy \n");
            for (int m = 2; m<=57; m++){
                int featIndex[] = pickNRandom(m);                              // pick m random indices from 1 to 57
                runSVM(featIndex,out);                                         // runSVM ( train and test) on tehse m features.
            }
        }
    }
    
    private static svm_problem getMainProblem() {
        svm_problem prob = new svm_problem();
        int numTrainExamples = featuresTraining.keySet().size();
        prob.l = numTrainExamples;
        prob.x = new svm_node[prob.l][];                                          // initialise SVM problem (training data)
        prob.y = new double[prob.l];
        
        for(int i =0; i<numTrainExamples; i++){
            HashMap<Integer, Double> temp = featuresTraining.get(i);               // get an example from featuresTraining HashMap
            prob.x[i]= new svm_node[temp.keySet().size()];                         // initialise svm_node
            int index =0;                                                          // index to keep track of feature number, index = id-1
            for (Integer id:temp.keySet()){                                        // id denotes the feature number 
                svm_node node= new svm_node();
                node.index = id;
                node.value = temp.get(id);                                         // assign feature index and value to the svm node
                prob.x[i][index] = node;
                index++;
            }
            prob.y[i]= labelTraining.get(i);                                       // get the label corresponding to the training example.
        }
        return prob;               

    }
    /*
    This method runs the kFold Validation algorithm. It divides the training set into kFold approximately equal sized , mutually disjoint subsets 
    and uses these sets as training and validation sets depending on the value of parameter 'c' of SVM.
    */
private static double runkFoldValidation(svm_parameter param, int lengthMain) {
        double accuracy[]= new double[kFold];
        // create hashmaps to contain the inputs and labels for the kFold sets.
        HashMap<Integer, HashMap<Integer, Double>>[] featNew= new HashMap[kFold];
        HashMap<Integer, Integer>[] labelSet = new HashMap[kFold];
        //get the approximate number of training examples in each set.
        int quotient= (int)lengthMain/kFold;
        int remainder = lengthMain%kFold;
        int length;
        for (int i=0; i<kFold; i++){
            if (i<remainder)
                length = quotient+1;
            else length= quotient;
            //get the 'length' input and labels.
            featNew[i] = getSet(length);
            labelSet[i]=getLabel(length);
            //increment global variable lengthTillNow, to keep track of examples already assigned to a set
            lengthTillNow += length;
        }
        
        for (int c = 0; c<kFold; c++){
            param.C = (double)(c+1)/kFold;

            svm_problem prob1 = getProblem(c,featNew,labelSet);                   // get the svm problem consisting of all training examples except Sc.
            svm_model model =  svm.svm_train(prob1, param);                       // train with given 'C' and the problem.

            //test the model on the left out set Sc.
            int accuracySum=0;
            for (Integer testInstance:featNew[c].keySet()){
                HashMap<Integer, Double> temp = featNew[c].get(testInstance);
                int numFeatures = temp.keySet().size();
                svm_node[] x = new svm_node[numFeatures];
                int featureIndex =0;
                for (Integer feature:temp.keySet()){               // for each feature of the test instance, get the index and value
                    x[featureIndex] = new svm_node();
                    x[featureIndex].index = feature;
                    x[featureIndex].value = temp.get(feature);
                    featureIndex++;
                }
                double d = svm.svm_predict(model,x);               // test SVM on this instance and get the predicted class d
                if (d==labelSet[c].get(testInstance))
                    accuracySum++;                                 // increment accuracy if correctly classified.
            }
        accuracy[c] = (double)accuracySum/featNew[c].size();
        }
        cParam= getBestC(accuracy);                               //get the c parameter which gives maximum accuracy on test data.
        return cParam;
    }
// This method executes similar to the testing in runkFoldValidation, except, it also calls functions to output data needed for ROC curve.
    private static void testSVM(svm_model model) throws IOException {
        int accuracySum=0;
        double finAccuracy;
        BufferedWriter out = new BufferedWriter (new FileWriter("test/outTemp.txt"));
        out.write("Actual Predicted  Val \n");
        for (Integer testInstance:featuresTesting.keySet()){
            HashMap<Integer, Double> temp = featuresTesting.get(testInstance);
            int numFeatures = temp.keySet().size();
            svm_node[] x = new svm_node[numFeatures];
            int featureIndex =0;
            for (Integer feature:temp.keySet()){
                x[featureIndex] = new svm_node();
                x[featureIndex].index = feature;
                x[featureIndex].value = temp.get(feature);
                featureIndex++;
            }
            double tar[] = new double[1];                     // to store decision values needed for calculating threshold, precision and recall
            double d = svm.svm_predict_values(model,x,tar);
            out.append(labelTesting.get(testInstance) + "     " +d +"       " + tar[0]+"\n");
         
            if (d==labelTesting.get(testInstance))
                accuracySum++;
        }
        finAccuracy = (double)accuracySum/labelTesting.size();
        System.out.println("accuracy= " +finAccuracy);
        out.close();
    }
    // This method returns a subset of training data , used in kFold cross validation
    private static HashMap<Integer, HashMap<Integer, Double>> getSet(int length) {
        HashMap<Integer, HashMap<Integer, Double>> featSet = new HashMap<>();
        for(int i=lengthTillNow; i<length+lengthTillNow; i++){
            featSet.put(i-lengthTillNow,featuresTraining.get(i));
        }
        return featSet;
    }
    // This method returns a subset of training data labels, used in kFold cross validation
    private static HashMap<Integer, Integer> getLabel(int length) {
         HashMap<Integer, Integer> featSet = new HashMap<>();
        for(int i=lengthTillNow; i<length+lengthTillNow; i++){
            featSet.put(i-lengthTillNow,labelTraining.get(i));
        }
        return featSet;
    }
    
    private static void runThresholdCurve() throws IOException {
        BufferedWriter  conf= new BufferedWriter(new FileWriter("test/confMatFile.txt"));        // file to print the output
        BufferedWriter conf2 = new BufferedWriter(new FileWriter ("test/PrecRecall.txt"));
        BufferedReader out = new BufferedReader(new FileReader("test/outTemp.txt"));
        Formatter fmt1 = new Formatter(conf2);
        double min =  -50.0;
        double max = 93;
        double spacing = (max-min)/200;
        double threshold = 0;
        fmt1.format("%12s   %12s   %12s   %12s\n", "threshold","precision","recall","fpr");
        for (threshold =min; threshold<=max; threshold += spacing){                             // vary threshold from min to max , incrementing it by spacing 
                                                                                               //and calculate precision and recall for each of these thresholds
            conf.write(" threshold= "+ threshold+"\n");
            BufferedWriter predict = new BufferedWriter(new FileWriter("test/outPredict.txt"));
            predict.write("Actual Predicted \n");
            String line;
            int testInstance = 0, d=-1 ;
            out.readLine();
            while((line = out.readLine()) != null){
                String str[]= line.split("\\s+");
                testInstance = Integer.parseInt(str[0]);
                double val = Double.parseDouble(str[1]);
                if (val>= threshold)
                    d = 1;
            predict.append(testInstance +"      "+ d);
            }
            int confMat[][] = getConfMat("test/outTemp.txt", threshold);
            
            printConfMat(conf, confMat);
            printPrecRecall(confMat,threshold,fmt1);
            }    conf.close();
        conf2.close();
    }

     private static int[][] getConfMat(String testoutTemptxt, double threshold) {
        int confMat[][] = new int [2][2];
        int i,j;           
        try {
            BufferedReader outFile = new BufferedReader(new FileReader(testoutTemptxt));  // file reader for out.txt
            String line=null;
            outFile.readLine();                                                          // ignore the title line
            while((line = outFile.readLine())!=null){                                    // while there is data to read in out.txt
                String[] str = line.split("\\s+");                                     // split the characters based on whitespaces
                 if(Integer.parseInt(str[0])==1)
                     i=0;
                 else i=1;   // i is actual, j is predicted
                 if(Double.parseDouble(str[2])>threshold)
                     j=0;
                 else j=1;
                confMat[i][j]= confMat[i][j]+1;                                                     // increment the value of the cell corresponding to actual character with index k and predicted character with index j
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        return confMat;
    }
     private static void printConfMat(BufferedWriter conf, int[][] confMat) {
        Formatter fmt = new Formatter(conf);                                      // to print a formatted array                        
        fmt.format("%4s", " ");  
        fmt.format("%5s  %5s\n", "pos", "neg");
        for (int i=0; i<2; i++){
            String str;
            if (i==0) 
                str="pos ";
            else str="neg ";
            fmt.format("%4s",str);
            for (int j=0; j<2; j++){
                fmt.format("%5d",confMat[i][j]);
            }
            fmt.format("\n");
        }        
    }

     private static void printPrecRecall(int[][] confMat, double threshold, Formatter fmt) {
        double precision,recall,fpr;
        int tp, fp, tn, fn;
        tp = confMat[0][0];
        fp = confMat[1][0];
        tn = confMat[1][1];
        fn = confMat[0][1];
        precision = (double)tp/(tp +fp);
        recall = (double)tp/(tp+fn);
        fpr = (double)fp/(fp+tn);
        fmt.format("%12f   %12f   %12f    %12f\n", threshold,precision,recall,fpr);
    }

     
    static double[] getWeights(svm_model model) throws IOException{
        double weights[] = new double [57];
        for (int j=0; j<57; j++){
            for (int i=0; i<model.nSV.length; i++){
                weights[j] += model.sv_coef[0][i] * model.SV[i][j].value;                // calculate weight vector using coefficients and support vectors
            }
        }
        for (int j =0; j<57; j++){
            weights[j] = abs(weights[j]);
        }
        return weights;
    }
    
    static Integer[] sortWeights(double[] weights){
        ArrayIndexComparator comparator = new ArrayIndexComparator(weights);
        Integer[] indexes = comparator.createIndexArray();
        Arrays.sort(indexes, comparator);
        return indexes;
    }
    
    static int[] getMlargestIndices(Integer[] ind, int m){
        int indices[] = new int[m];
        for (int i=0; i<m; i++){
            indices[i] = ind[56-i];                    // return m elements from end of an ascending-sorted array
        }
        return indices;
    }

    //train and test SVM based on the features dictated by featIndex
    static void runSVM(int[] featIndex, BufferedWriter out) throws IOException{
        svm_problem prob = getMProblem(featIndex);
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.LINEAR;
        param.C = cParam;
        svm_model model =svm.svm_train(prob, param) ;
        testMSVM(model, featIndex,out);
        
    }
    
    private static  svm_problem getMProblem(int[] featIndex) {
        svm_problem prob = new svm_problem();
        int numTrainExamples = featuresTraining.keySet().size();
        prob.l = numTrainExamples;
        prob.x = new svm_node[prob.l][];
        prob.y = new double[prob.l];
        
        for(int i =0; i<numTrainExamples; i++){
            HashMap<Integer, Double> temp = featuresTraining.get(i);
            prob.x[i]= new svm_node[featIndex.length];
            int index =0;
            for (Integer id:featIndex){
                svm_node node= new svm_node();
                node.index = index;
                node.value = temp.get(id);
                prob.x[i][index] = node;
                index++;
            }
            prob.y[i]= labelTraining.get(i);
        }
        return prob;
    }

    private static void testMSVM(svm_model model, int[] featIndex, BufferedWriter out ) throws IOException {
         int accuracySum=0;
            double finAccuracy;
            
            for (Integer testInstance:featuresTesting.keySet()){
                HashMap<Integer, Double> temp = featuresTesting.get(testInstance);//new HashMap<Integer, Double>();
                svm_node[] x = new svm_node[featIndex.length];
                int featureIndex =0;
                for (Integer feature:featIndex){
                    x[featureIndex] = new svm_node();
                    x[featureIndex].index = featureIndex;
                    x[featureIndex].value = temp.get(feature);
                    featureIndex++;
                }
                int d = (int)svm.svm_predict(model,x);
                if (d==labelTesting.get(testInstance))
                    accuracySum++;
            }
        finAccuracy = (double)accuracySum/labelTesting.size();
        out.append(featIndex.length + "   "+ finAccuracy+ "\n");
    }

    private static svm_problem getProblem(int c,HashMap<Integer, HashMap<Integer, Double>>[]feat, HashMap<Integer,Integer>[] lab) {
        svm_problem prob = new svm_problem();
        int length = 0;
        for (int i=0;i<feat.length;i++){
            if (c!=i)
                length +=feat[i].size();
        }
        prob.l = length;
        prob.x = new svm_node[prob.l][];
        prob.y = new double[prob.l];
        int xIndex =0;
        for (int k=0;k<feat.length;k++){
            if (c!=k){
                for (int i=0; i<feat[k].size(); i++){
                HashMap<Integer, Double> temp = feat[k].get(i);
                prob.x[xIndex]= new svm_node[temp.keySet().size()];
                int index =0;
                for (Integer id:temp.keySet()){
                    svm_node node= new svm_node();
                    node.index = id;
                    node.value = temp.get(id);
                    prob.x[xIndex][index] = node;
                    index++;
                    
                }
                prob.y[xIndex]= lab[k].get(i);
                xIndex++;
                }
            }   
        }
        return prob;
       }
    
    
    /**
     * @param args the command line arguments
     */
    

    
    private static double getBestC(double[] accuracy) {
        int max = 0;
        for (int i=0; i<accuracy.length; i++){
            if(accuracy[i]>accuracy[max])
                max=i;
            System.out.println("c = "+ ((double)(i+1)/kFold) + "  accuracy = " +accuracy[i]);
        }
        return (double)(max+1)/kFold;
    }

    static int[] pickNRandom(int n){
       int[] lst = IntStream.range(1,57).toArray();
        List <Integer> list = new ArrayList<>(lst.length);
        for (int i: lst)
            list.add(i);
        Collections.shuffle(list);
        int[] ans = new int[n];
        for (int i=0; i<n; i++)
            ans[i] = list.get(i);
        return ans;
    }
}
