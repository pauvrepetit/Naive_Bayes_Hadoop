import java.io.*;
import java.util.*;

class PriorProbability {
    private final String classCountFilename;
    private final Map<String, Double> classCount = new HashMap<>();
    private int allDocumentCount = 0;

    public PriorProbability(String classCountFilename) throws IOException {
        this.classCountFilename = classCountFilename;
        this.loadClassCount();
        this.computePriorProbability();
    }

    /**
     * 加载classCount文件，将数据保存到classCount这个map中
     */
    private void loadClassCount() throws IOException {
        File file = new File(this.classCountFilename);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        String[] ss;
        int count;
        while ((line = reader.readLine()) != null) {
            // 读一行，将其拆分为<类型名，文档数量>，同时统计所有文档的总数
            ss = line.split("\t");
            count = Integer.parseInt(ss[1]);
            classCount.put(ss[0], (double) count);
            this.allDocumentCount += count;
        }
        reader.close();
    }

    /**
     * 计算先验概率，也就是classCount中的每一个count除以count的总和
     */
    private void computePriorProbability() {
        classCount.replaceAll((k, v) -> v / allDocumentCount);
    }

    public double get(String className) {
        return classCount.get(className);
    }

    /**
     * 获取所有的类型，组成的集合
     *
     * @return 类型集合
     */
    public Set<String> getClassNames() {
        return classCount.keySet();
    }
}

class ConditionalProbability {
    private final String classTermFilename;
    private final Map<String, Map<String, Double>> condProbability = new HashMap<>();
    private final Map<String, Double> classTermCount = new HashMap<>();   // 保存每个类型class对应的所有文档中的term总数
    private int allTermCount; // 统计term的种类数
    private final Set<String> terms = new HashSet<>();


    public ConditionalProbability(String classTermFilename) throws IOException {
        this.classTermFilename = classTermFilename;
        this.loadClassTerm();
        this.computeCondProbability();
    }

    private void loadClassTerm() throws IOException {
        File file = new File(this.classTermFilename);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        String[] ss;
        String className, term;
        int count;
        Map<String, Double> termCount;
        while ((line = reader.readLine()) != null) {
            // 读一行，<类型名c，单词t，单词t在所有类型为c的文档中出现的总次数>
            ss = line.split("\t");
            className = ss[0];
            term = ss[1];
            count = Integer.parseInt(ss[2]);

            terms.add(term);
            if (condProbability.containsKey(className)) {
                termCount = condProbability.get(className);
            } else {
                termCount = new HashMap<>();
            }
            termCount.put(term, (double) count);
            condProbability.put(className, termCount);
            allTermCount = terms.size();
        }
        reader.close();

        for (Map.Entry<String, Map<String, Double>> entry : condProbability.entrySet()) {
            className = entry.getKey();
            double c = 0;
            for (Map.Entry<String, Double> entry1 : entry.getValue().entrySet()) {
                c += entry1.getValue();
            }
            classTermCount.put(className, c);
        }
    }

    /**
     * 计算似然概率，对于一个单词term和一个类别class，可以求一个似然概率P(term|class)
     * 分子是类别是class的文档中，单词term出现的次数，加1，就是condProbability中加载的那个值+1
     * 分母是类别是class的文档中，所有单词的总数，加单词的种类数(也就是所有的文档中，一共有多少种不同的单词)
     * 分母的前者，我们就是classTermCount中统计的值，后者，就是allTermCount的值
     */
    private void computeCondProbability() {
        for (Map.Entry<String, Map<String, Double>> entry : condProbability.entrySet()) {
            entry.getValue().replaceAll((k, v) -> (v + 1) / (classTermCount.get(entry.getKey()) + allTermCount));
        }
    }

    /**
     * 获取单词term在类型className上的似然概率
     *
     * @param className 类型
     * @param term      单词
     * @return 似然概率
     */
    public double get(String className, String term) {
        try {
            return condProbability.get(className).get(term);
        } catch (Exception e) {
            // 如果类型className的文档中没有出现过单词term，那么我们的condProbability中是不存在它对应的似然概率的
            // 此时，我们做一个单独的计算
            return 1.0 / (classTermCount.get(className) + allTermCount);
        }
    }
}

public class Prediction {
    private static Map<String, Integer> loadDocument(String filename) throws IOException {
        Map<String, Integer> docTerms = new HashMap<>();
        File file = new File(filename);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String term;
        while ((term = reader.readLine()) != null) {
            if (!docTerms.containsKey(term)) {
                docTerms.put(term, 1);
            } else {
                docTerms.replace(term, docTerms.get(term) + 1);
            }
        }
        reader.close();
        return docTerms;
    }

    /**
     * 计算后验概率的对数，我们需要求文档docTerms属于每一个类型的概率
     * 对于类型i，后验概率等于类型i的先验概率 乘以 每一个单词属于类型i的似然概率
     * 前者，在pp中取一个值即可，后者，对docTerms中的每个单词，结合类型i，在cp中取一个值
     * 为了保证浮点数计算不会向下溢出，我们对这个计算过程取对数，也就是把取出来的值都取对数，然后乘法就变成了加法
     *
     * @param docTerms 待分类的文档的内容信息，term，count，单词以及每个单词出现的次数
     * @param pp       先验概率，包含了各种类型信息，以及每种类型的概率
     * @param cp       似然概率，包含了类型、单词
     * @return 返回文档属于各种类型的后验概率
     */
    private static Map<String, Double> computePosteriorProbability(Map<String, Integer> docTerms, PriorProbability pp
            , ConditionalProbability cp) {
        Map<String, Double> posteriorProbability = new HashMap<>();
        Set<String> classNames = pp.getClassNames();
        for (String className : classNames) {
            double postP = Math.log(pp.get(className));   // 第一部分，类型className的先验概率
            for (Map.Entry<String, Integer> entry : docTerms.entrySet()) {
                double p = cp.get(className, entry.getKey());
                postP += Math.log(p) * entry.getValue();
            }
            posteriorProbability.put(className, postP);
        }
        return posteriorProbability;
    }

    /**
     * 预测文档的类型
     */
    private static String predict(String filename, PriorProbability pp, ConditionalProbability cp) throws IOException {
        Map<String, Integer> docTerms = loadDocument(filename);
        Map<String, Double> postP = computePosteriorProbability(docTerms, pp, cp);
        String className = null;
        double probability = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> entry : postP.entrySet()) {
            if (entry.getValue() > probability) {
                className = entry.getKey();
                probability = entry.getValue();
            }
        }
        return className;
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            // 输入3个参数，先验概率、似然概率、待预测文档
            System.out.println("args error. Prediction <classCountOutput> <classTermOutput> <document dir>");
            return;
        }
        String classCountFilename = args[0];    // 统计各类别的文档的数量的mapreduce任务的输出文件，用于计算先验概率
        String classTermFilename = args[1];     // 统计类别、单词出现次数的mapreduce任务的输出文件，用于计算似然概率
        String prePredictDocumentFilenameDir = args[2];    // 待预测的文档所在的目录

        PriorProbability priorProbability = new PriorProbability(classCountFilename);
        ConditionalProbability conditionalProbability = new ConditionalProbability(classTermFilename);

        File superDir = new File(prePredictDocumentFilenameDir);
        File[] subDirs = superDir.listFiles();
        assert subDirs != null;
        int classCount = subDirs.length;    // 类型数量，一共有classCount种不同的类型
        int docCount = 0;   // 文档总数
        int[] tp = new int[classCount];     // 对于某一种类型class，文档真实类型为class且预测类型为class的数量
        //int[] tn = new int[classCount];     // 对于某一种类型class，文档真实类型为class且预测类型不为class的数量
        int[] fp = new int[classCount];     // 对于某一种类型class，文档真实类型不为class且预测类型为class的数量
        int[] fn = new int[classCount];     // 对于某一种类型class，文档真实类型不为class且预测类型不为class的数量
        double[] precision = new double[classCount];
        double[] recall = new double[classCount];
        double[] f1 = new double[classCount];
        String[] classNames = new String[classCount];   // 类型名数组
        for (int i = 0; i < classCount; i++) {
            classNames[i] = subDirs[i].getName();
            docCount += Objects.requireNonNull(subDirs[i].listFiles()).length;
        }
        String[] docs = new String[docCount];
        String[] docClass = new String[docCount];
        String[] predictClass = new String[docCount];

        int count = 0;
        for (int i = 0; i < classCount; i++) {
            File[] subDocs = subDirs[i].listFiles();
            assert subDocs != null;
            for (File subDoc : subDocs) {
                docs[count] = subDoc.toString();
                docClass[count] = classNames[i];
                predictClass[count] = predict(docs[count], priorProbability, conditionalProbability);
                count++;
            }
        }

        for (int i = 0; i < classCount; i++) {
            tp[i] = fn[i] = fp[i] = 0;
            for (int j = 0; j < docCount; j++) {
                boolean b1 = docClass[j].equals(classNames[i]);
                boolean b2 = predictClass[j].equals(classNames[i]);
                if (b1 && b2) {
                    tp[i]++;
                } else if (b1) {
                    fn[i]++;
                } else if (b2) {
                    fp[i]++;
                }

            }
            precision[i] = tp[i] * 1.0 / (tp[i] + fp[i]);
            recall[i] = tp[i] * 1.0 / (tp[i] + fn[i]);
            f1[i] = 2.0 * precision[i] * recall[i] / (precision[i] + recall[i]);
        }

        double micro_precision = 0, micro_recall = 0, micro_f1 = 0;
        int all_tp = 0, all_fp = 0, all_fn = 0;
        double macro_precision, macro_recall, macro_f1;

        for (int i = 0; i < classCount; i++) {
            micro_precision += precision[i];
            micro_recall += recall[i];
            micro_f1 += f1[i];

            all_tp += tp[i];
            all_fp += fp[i];
            all_fn += fn[i];
        }

        micro_precision /= classCount;
        micro_recall /= classCount;
        micro_f1 /= classCount;

        macro_precision = all_tp * 1.0 / (all_tp + all_fp);
        macro_recall = all_tp * 1.0 / (all_tp + all_fn);
        macro_f1 = 2.0 * macro_precision * macro_recall / (macro_precision + macro_recall);

        System.out.println("prediction result:");
        System.out.println("document\ttrueClass\tpredictClass");
        for (int i = 0; i < docCount; i++) {
            System.out.printf("%s\t%s\t%s\n", docs[i], docClass[i], predictClass[i]);
        }

        System.out.printf("micro: precision is %f, recall is %f, f1 is %f\n", micro_precision, micro_recall,
                micro_f1);
        System.out.printf("macro: precision is %f, recall is %f, f1 is %f\n", macro_precision, macro_recall,
                macro_f1);
    }
}
