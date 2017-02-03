/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multineuralnetwork;

/**
 *
 * @author VidhyaLakshmi
 */
public class Result {
    double hiddenAct[];
    double output[];
    double input[];
    Result(int inputs,int hiddenLayers, int outputs){
        this.hiddenAct = new double[hiddenLayers];
        this.output = new double[outputs];
        this.input = new double[inputs];
    }
    
}
