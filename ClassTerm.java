import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * 两个Text组成的pair，作为key的类型，<Classname, Term>对
 */
class TextPair implements WritableComparable<TextPair> {
    private Text first;
    private Text second;

    public TextPair() {
        set(new Text(), new Text());
    }

    public TextPair(String first, String second) {
        set(new Text(first), new Text(second));
    }

    public TextPair(Text first, Text second) {
        set(first, second);
    }

    public void set(Text first, Text second) {
        this.first = first;
        this.second = second;
    }

    public Text getFirst() {
        return first;
    }

    public Text getSecond() {
        return second;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        first.write(out);
        second.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        first.readFields(in);
        second.readFields(in);
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof TextPair) {
            TextPair tp = (TextPair) o;
            return first.equals(tp.first) && second.equals(tp.second);
        }
        return false;
    }

    @Override
    public String toString() {
        return first + "\t" + second;
    }

    public int compareTo(TextPair tp) {
        int cmp = first.compareTo(tp.first);
        if (cmp != 0) {
            return cmp;
        }
        return second.compareTo(tp.second);
    }
}

class ClassTermRecordTeader extends RecordReader<TextPair, IntWritable> {

    FileSplit split;
    Text classname = new Text();
    Text term = new Text();
    TextPair class_term_pair = new TextPair();
    IntWritable one = new IntWritable(1);
    Configuration conf;

    LineRecordReader lineRecordReader;

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
        this.split = (FileSplit) split;

        String[] ss = this.split.getPath().toString().split("/");
        classname.set(ss[ss.length - 2]);

        this.conf = context.getConfiguration();
        // 我们直接使用一个lineRecordReader来读取文件，而不需要我们自己手动操作
        // 我们只需要对它输出的K-V对进行进行一定的调整，向其中添加一个类型名信息即可，这个我们在getCurrentKey中完成即可
        lineRecordReader = new LineRecordReader();
        lineRecordReader.initialize(split, context);
    }

    /**
     * 判断是否还有输入需要读取，如果有，就返回true，此时hadoop将调用getCurrentKey和getCurrentValue来获取<key,value>对，并将其传递给map函数
     * 如果输入全部读取完成，返回false，当前map任务的输入结束
     * 
     * 我们直接从lineRecordReader中获取文件读取状态
     */
    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        return lineRecordReader.nextKeyValue();
    }

    /**
     * 返回的<key,value>对中的key，它是一个由类型名classname和分词term组成的Pair
     */
    @Override
    public TextPair getCurrentKey() throws IOException, InterruptedException {
        term.set(lineRecordReader.getCurrentValue());
        class_term_pair.set(classname, term);
        return class_term_pair;
    }

    /**
     * 返回的<key,value>对中的value，为常数1
     */
    @Override
    public IntWritable getCurrentValue() throws IOException, InterruptedException {
        return this.one;
    }

    /**
     * 返回当前输入的进度，返回一个进度的百分比
     * 由于我们直接使用lineRecordReader来对文件进行读取，我们的一个map任务只处理这一个lineRecordReader，因此可以直接使用它的progress作为我们的map任务的progress
     */
    @Override
    public float getProgress() throws IOException, InterruptedException {
        return lineRecordReader.getProgress();
    }

    @Override
    public void close() throws IOException {
    }
}

/**
 * ClassTermInputFormat
 * map的输入数据的类型，通过recordReader不断获取输入的<key,value>对
 */
class ClassTermInputFormat extends FileInputFormat<TextPair, IntWritable> {

    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

    @Override
    public ClassTermRecordTeader createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException, InterruptedException {
        ClassTermRecordTeader recordReader = new ClassTermRecordTeader();
        recordReader.initialize(split, context);
        return recordReader;
    }
}

public class ClassTerm {

    public static class TokenizerMapper extends Mapper<TextPair, IntWritable, TextPair, IntWritable> {

        /**
         * map函数，直接将输入的<<classname,term>,1>对向后传递，不需要进行其他操作
         */
        public void map(TextPair key, IntWritable value, Context context) throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    public static class IntSumReducer extends Reducer<TextPair, IntWritable, TextPair, IntWritable> {
        private IntWritable sum_value = new IntWritable();

        /**
         * reduce函数，对于输入的<<classname,term>,{value1,value2...}>，将所有的value相加，得到value，输出<<classname,term>,value>，classname是类型名，term是分词，value是在类型classname的文档中分词term出现的总次数
         */
        public void reduce(TextPair key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            sum_value.set(sum);
            context.write(key, sum_value);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length < 2) {
            System.err.println("Usage: ClassTerm <in> [<in>...] <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "ClassTerm");
        job.setJarByClass(ClassTerm.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setInputFormatClass(ClassTermInputFormat.class);

        job.setOutputKeyClass(TextPair.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.setInputDirRecursive(job, true);
        for (int i = 0; i < otherArgs.length - 1; ++i) {
            System.out.println(otherArgs[i]);
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job,
                new Path(otherArgs[otherArgs.length - 1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
