package com.user.consumer.CodeGenerater;


import com.baomidou.mybatisplus.core.toolkit.StringPool;
import com.baomidou.mybatisplus.generator.AutoGenerator;
import com.baomidou.mybatisplus.generator.InjectionConfig;
import com.baomidou.mybatisplus.generator.config.*;
import com.baomidou.mybatisplus.generator.config.converts.MySqlTypeConvert;
import com.baomidou.mybatisplus.generator.config.po.TableInfo;
import com.baomidou.mybatisplus.generator.config.rules.DbColumnType;
import com.baomidou.mybatisplus.generator.config.rules.IColumnType;
import com.baomidou.mybatisplus.generator.config.rules.NamingStrategy;
import com.baomidou.mybatisplus.generator.engine.FreemarkerTemplateEngine;

import java.util.ArrayList;
import java.util.List;

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
//        String projectPath = System.getProperty("E:/gitHubDevHome/robotization/user-server/user-server-consumer");
        String path = "E:/gitHubDevHome/robotization/user-server/user-server-consumer";
        // 生成文件的输出目录，默认D盘根目录
        gc.setOutputDir(path+"/src/main/java");
        // 是否覆盖已有文件
        gc.setFileOverride(true);
        // 开发人员
        gc.setAuthor("cq");
        // 是否打开输出目录,默认true
        gc.setOpen(false);
        // 否在xml中添加二级缓存配置,默认false
        gc.setEnableCache(false);
        // 开启 swagger2 模式,默认false，这里推荐使用JApiDocs，使用方法见：https://blog.csdn.net/u013919153/article/details/110440311
        gc.setSwagger2(false);
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
        DataSourceConfig dsc = new DataSourceConfig();
        dsc.setUrl("jdbc:mysql://127.0.0.1:3306/c_robotization?serverTimezone=UTC&useUnicode=true&characterEncoding=utf-8&useSSL=false");
        dsc.setSchemaName("databaseName");
        dsc.setDriverName("com.mysql.jdbc.Driver");
        dsc.setUsername("root");
        dsc.setPassword("root");
        dsc.setTypeConvert(new MySqlTypeConvert());
        auto.setDataSource(dsc);

        // 包名
        PackageConfig pc = new PackageConfig();
//        pc.setModuleName("user-server-consumer");
        pc.setParent("com.user.consumer");
        pc.setEntity("entity");
        pc.setMapper("dao");
        pc.setController("Controller");
        auto.setPackageInfo(pc);

        // 自定义配置
        InjectionConfig cfg = new InjectionConfig() {
            @Override
            public void initMap() {
            }
        };

        // 模板引擎
        String templatePath = "/templates/mapper.xml.ftl";

        // 自定义输出配置
        List<FileOutConfig> focList= new ArrayList<>();
        focList.add(new FileOutConfig() {
            @Override
            public String outputFile(TableInfo tableInfo) {
                // 自定义xml文件输出名
                return path + "/src/main/java/com/user/consumer/mapper" + tableInfo.getEntityName() + "Mapper" + StringPool.DOT_XML;
            }
        });
         cfg.setFileOutConfigList(focList);
         auto.setCfg(cfg);

         // 配置模板
        TemplateConfig templateConfig = new TemplateConfig();
        // 配置自定义输出模板
//        templateConfig.setEntity();
//          templateConfig.setService();
            templateConfig.setXml(null);
            templateConfig.setController(null);
            auto.setTemplate(templateConfig);

        // 策略配置
        StrategyConfig strategy = new StrategyConfig();
        // 数据库表映射到实体的命名策略
        strategy.setNaming(NamingStrategy.underline_to_camel);
        // 数据库表字段映射到实体的命名策略, 未指定按照 naming 执行
        strategy.setColumnNaming(NamingStrategy.underline_to_camel);
        //自定义继承的Entity类全称，带包名
        //strategy.setSuperEntityClass("com.apidoc.demo.Entity.BaseEntity");
        // 自定义基础的Entity类，公共字段
        //strategy.setSuperEntityColumns(new String[] {"id","gmtCreate","gmtModified"});
        // 是否为lombok模型
        strategy.setEntityLombokModel(true);
        // Boolean类型字段是否移除is前缀
        strategy.setEntityBooleanColumnRemoveIsPrefix(true);
        // 生成 @RestController 控制器
        strategy.setRestControllerStyle(false);
        //strategy.setSuperControllerClass("com.music.taosim.ant.common.BaseController");
        // 当对某张表有所改动但只想重新生成这张表，可以这样设置
        //startegy.setInclude("tableName");
        // 驼峰转连字符 如 umps_user 变为 upms/user
        strategy.setControllerMappingHyphenStyle(true);
        // 表前缀
        // strategy.setTablePrefix(pc.getModuleName() + "_");
        auto.setStrategy(strategy);
        //设置模板引擎类型，默认为 velocity
        auto.setTemplateEngine(new FreemarkerTemplateEngine());
        auto.execute();
    }
}
/**
 * 自定义类型转换 ,解决mybatis plus自动生成代码tinyint(1)自动转换为Boolean
 */
class MySqlTypeConvertCustom extends MySqlTypeConvert implements ITypeConvert {
    @Override
    public IColumnType processTypeConvert(GlobalConfig globalConfig, String fieldType) {
        String t = fieldType.toLowerCase();
        if (t.contains("tinyint(1)")) {
            return DbColumnType.INTEGER;
        }
        return super.processTypeConvert(globalConfig, fieldType);
    }
}