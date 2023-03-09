package CodeGenerater;


import com.baomidou.mybatisplus.generator.AutoGenerator;
import com.baomidou.mybatisplus.generator.config.GlobalConfig;

/**
 * 代码生成器
 */
public class CodeGenerater {
    public static void main(String[] args) {
        // 代码生成器
        AutoGenerator auto = new AutoGenerator();
        // 全局配置
        GlobalConfig gc = new GlobalConfig();
        // 项目名和路径
        String project = System.getProperty("user-server-consumer");
        // 生成文件的输出目录，默认D盘根目录
        gc.setOutputDir(project+"/src/main/java");
        // 是否覆盖已有文件
        gc.setFileOverride(true);
        // 开发人员
        gc.setAuthor("cq");
        // 是否打开输出目录,默认true
        gc.setOpen(false);
        // 否在xml中添加二级缓存配置,默认false
        gc.setEnableCache(false);
        // 开启 swagger2 模式,默认false，这里推荐使用JApiDocs，使用方法见：https://blog.csdn.net/u013919153/article/details/110440311
        gc.setSwagger2(true);
        //指定生成实体类名称
        gc.setEntityName("%s");
        // mapper文件名称
        gc.setMapperName("%sMapper");
        // xml文件名称
        gc.setXmlName("%sMapper");
        // 服务名称
        gc.setServiceName("I%sService");
        // 服务接口名称
        gc.setServiceImplName("%sServiceImpl");
        auto.setGlobalConfig(gc);

        // 数据源配置

    }
}
