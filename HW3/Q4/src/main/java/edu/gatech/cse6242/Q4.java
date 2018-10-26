package edu.gatech.cse6242;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;

public class Q4 {

  //First Mapper
  public static class FirstMapper
  extends Mapper<Object, Text, Text, IntWritable>{
    private Text source = new Text();
    private Text target = new Text();
    private final static IntWritable degree_count = new IntWritable(1);
    private final static IntWritable negdegree_count = new IntWritable(-1);

    public void map(Object key, Text value, Context context
      ) throws IOException, InterruptedException {
      String lines = value.toString();
      if ((lines != null) && (!lines.equals(""))) {
        String [] line = lines.split("\\t");
        source.set(line[0]);
        target.set(line[1]);
        context.write(source, degree_count);
        context.write(target, negdegree_count);
      }
    }
  }

  //First Reducer
  public static class FirstReducer
  extends Reducer<Text, IntWritable, Text, IntWritable>{
    private IntWritable diff = new IntWritable();
    public void reduce(Text key, Iterable<IntWritable> values, Context context
      ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      diff.set(sum);
      context.write(key, diff);
    }
  }

  //Second Mapper
  public static class SecondMapper
  extends Mapper<Object, Text, Text, IntWritable>{
    private Text diff = new Text();
    private final static IntWritable final_count = new IntWritable(1);

    public void map(Object key, Text value, Context context
      ) throws IOException, InterruptedException {
      String lines = value.toString();
      String [] line = lines.split("\\t");
      diff.set(line[1]);
      context.write(diff, final_count);
    }
  }

  //Second Reducer
  public static class SecondReducer
  extends Reducer<Text, IntWritable, Text, IntWritable>{
    private IntWritable final_count = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context
      ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();  // Calculate the count of the diff with same key
      }
      final_count.set(sum);
      context.write(key, final_count);
    }
  }


  public static void main(String [] args) throws Exception {
    Configuration conf1 = new Configuration();
    Job job1 = Job.getInstance(conf1, "Q4_1");
    
    job1.setJarByClass(Q4.class);
    FileInputFormat.addInputPath(job1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job1, new Path(args[1]+"temp"));

    job1.setMapperClass(FirstMapper.class);
    job1.setReducerClass(FirstReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(IntWritable.class);
    job1.waitForCompletion(true);

    Configuration conf2 = new Configuration();
    Job job2 = Job.getInstance(conf2, "Q4_2");

    job2.setJarByClass(Q4.class);
    FileInputFormat.addInputPath(job2, new Path(args[1]+"temp"));
    FileOutputFormat.setOutputPath(job2, new Path(args[1]));

    job2.setMapperClass(SecondMapper.class);
    job2.setReducerClass(SecondReducer.class);
    job2.setOutputKeyClass(Text.class);
    job2.setOutputValueClass(IntWritable.class);

    System.exit(job2.waitForCompletion(true) ? 0 : 1);
  }
}
