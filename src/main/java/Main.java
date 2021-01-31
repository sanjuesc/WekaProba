import weka.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import javax.swing.plaf.nimbus.NimbusLookAndFeel;
import java.util.Random;

public class Main {
    public static void main (String [ ] args) throws Exception {

        ConverterUtils.DataSource source = new ConverterUtils.DataSource("heart-c.arff");
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        System.out.println(data.numAttributes());
        System.out.println(data.numInstances());
        System.out.println(data.attribute(0));
        System.out.println(data.attributeStats(0));
        NaiveBayes c = new NaiveBayes();
        data.setClassIndex(13);
        c.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(c, data, 3, new Random());
        eval.confusionMatrix();
        System.out.println(eval.toMatrixString("num"));
        System.out.println(eval.errorRate());
        System.out.println("-----------------");
        System.out.println(eval.weightedPrecision());

        for(int a = 0; a<5;a++){
            System.out.println(eval.precision(a));
        }



    }

}
