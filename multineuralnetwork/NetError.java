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
public class NetError {
    double dk[];
    double dj[];
    NetError(int hiddenLayers){
        this.dk = new double[26];
        this.dj = new double[hiddenLayers];
    }
}
