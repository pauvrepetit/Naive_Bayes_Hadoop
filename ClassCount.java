import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

class ClassnameRecordReader extends RecordReader<Text, IntWritable> {

    FileSplit split;
    Text classname = new Text();
    IntWritable one = new IntWritable(1);
    Configuration conf;
    int count = 0;      // 这是当前访问到的文件数量
    int all_count = 1;  // 这是待访问的文件总数
    boolean finish = false;

    FileStatus[] files;

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
        this.split = (FileSplit) split;
        this.conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);     // 获取文件系统，后续的文件访问操作都在fs上完成
        Path path = this.split.getPath();         // 获取路径，这是输入目录下的每一个文件(文件或者子目录)
        files = fs.listStatus(path);              // 获取该路径下的所有文件信息
        // 我们可以保证的是，path必然是一个子目录，我们获取子目录下的所有文件的文件信息，然后遍历他们
        all_count = files.length;
    }

    /**
     * 判断是否还有输入需要读取，如果有，就返回true，此时hadoop将调用getCurrentKey和getCurrentValue来获取<key,value>对，并将其传递给map函数
     * 如果输入全部读取完成，返回false，当前map任务的输入结束
     * 
     * files是输入的一个目录下面的文件列表，我们在init中通过文件系统操作获取到的
     * 我们这里做的就是对于files中的每一个文件，输出一个<classname,1>的K-V对，我们不需要去具体的访问这些文件，只需要从文件路径中获取到其对应的类型信息即可
     */
    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if (count < all_count) {
            // 还有文件没有访问，返回true，设置classname
            String[] ss = files[count].getPath().toString().split("/");
            classname.set(ss[ss.length - 2]);
            count++;
            return true;
        } else {
            // 所有文件都访问完了，返回false
            finish = true;
            return false;
        }
    }

    /**
     * 返回的<key,value>对中的key，为当前文档的类型
     */
    @Override
    public Text getCurrentKey() throws IOException, InterruptedException {
        return this.classname;
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
     * 一个map任务就是遍历一个目录下的每个文件，我们知道目录下的文件总数为all_count，知道当前访问过的文件数量count，因此可以计算进度
     */
    @Override
    public float getProgress() throws IOException, InterruptedException {
        return this.count * 1.0f / this.all_count;
    }

    @Override
    public void close() throws IOException {
    }
}

/**
 * classInputFormat
 * map的输入数据的类型，通过recordReader不断获取输入的<key,value>对，我们要做的就是创建并返回一个我们自定义的recordReader对象
 */
class classInputFormat extends FileInputFormat<Text, IntWritable> {

    @Override
    protected boolean isSplitable(JobContext context, Path filename) {
        return false;
    }

    @Override
    public ClassnameRecordReader createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException, InterruptedException {
        ClassnameRecordReader recordReader = new ClassnameRecordReader();
        recordReader.initialize(split, context);
        return recordReader;
    }
}

public class ClassCount {

    public static class TokenizerMapper extends Mapper<Text, IntWritable, Text, IntWritable> {

        /**
         * map函数，直接将输入的<classname,1>对向后传递，不需要进行其他操作
         */
        public void map(Text key, IntWritable value, Context context) throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable sum_value = new IntWritable();

        /**
         * reduce函数，对于输入的<classname,{value1,value2...}>，将所有的value相加，得到value，输出<classname,value>
         * classname是一个类型名，value是该类型的文档出现的总次数
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
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
            System.err.println("Usage: ClassCount <in> [<in>...] <out>");
            System.exit(2);
        }
        Job job = Job.getInstance(conf, "ClassCount");
        job.setJarByClass(ClassCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setInputFormatClass(classInputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        for (int i = 0; i < otherArgs.length - 1; ++i) {
            System.out.println(otherArgs[i]);
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job,
                new Path(otherArgs[otherArgs.length - 1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
