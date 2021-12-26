# Naive Bayes on Hadoop

使用hadoop mapreduce实现一个贝叶斯分类器

有三个部分组成
1. ClassCount，统计训练集中包含的各种类型的文档的数量，用于计算先验概率
2. ClassTerm，统计某种类型的所有文档中，各单词出现的数量，用于计算条件概率
3. Prediction，根据上面两部分的输出，计算先验概率和条件概率，对文档进行分类

其中，ClassCount和ClassTerm是mapreduce程序，Prediction是普通的java程序

### 使用方法
``` sh
mkdir bin_class_count
javac -cp $(hadoop classpath) -d ./bin_class_count ./ClassCount.java
jar -cvf ./ClassCount.jar ./bin_class_count .

mkdir bin_class_term
javac -cp $(hadoop classpath) -d ./bin_class_term ./ClassTerm.java
jar -cvf ./ClassTerm.jar ./bin_class_term .

hadoop jar ./ClassCount.jar ClassCount /NBCorpus/Country /ClassCount_Output_Country
hadoop jar ./ClassTerm.jar ClassTerm /NBCorpus/Country /ClassTerm_Output_Country

hadoop jar ./ClassCount.jar ClassCount /NBCorpus/Industry /ClassCount_Output_Industry
hadoop jar ./ClassTerm.jar ClassTerm /NBCorpus/Industry /ClassTerm_Output_Industry

javac -d ./bin_prediction ./Prediction.java
cd bin_prediction
java Prediction ../data/ClassCount_Output_Country/part-r-00000 ../data/ClassTerm_Output_Country/part-r-00000 ../data/NBCorpus/Country
java Prediction ../data/ClassCount_Output_Industry/part-r-00000 ../data/ClassTerm_Output_Industry/part-r-00000 ../data/NBCorpus/Industry
```
