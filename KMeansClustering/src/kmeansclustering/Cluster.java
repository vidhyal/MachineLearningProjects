/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kmeansclustering;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author VidhyaLakshmi
 */

public class Cluster {
    double[] center;
    List<Integer> datapoints;
    double sSE ;
    int lab[];
    int numFeat =64;
    int label;
    
    Cluster(int seed){
        Random ran = new Random(seed);
        center = new double[numFeat];
        for (int j=0; j<numFeat ; j++){
            this.center[j] = 16*ran.nextDouble();
        }
        this.datapoints = new ArrayList<Integer>();    
        this.sSE =0;
        this.lab = new int[10];
    }
    public static Cluster copyCluster(Cluster cluster){
        Cluster cluster2 = new Cluster(0);
        for (int i=0; i<cluster.center.length; i++){
            cluster2.center[i] = cluster.center[i];
        }
        for (int j=0; j<cluster.datapoints.size(); j++){
            cluster2.datapoints.add(cluster.datapoints.get(j));
        }
        return cluster2;
        
    }
}
