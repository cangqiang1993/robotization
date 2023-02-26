package cn.wolfcode.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Created by lanxw
 */
@RestController
@RefreshScope
public class NacosConfigController {
    @Value("${appConfig.name}")
    private String appConfigName;
    @Value("${env}")
    private String env;
    @Value("${globalConfig}")
    private String globalConfig;
    @RequestMapping("/nacosConfig1")
    public String nacosConfig(){
        return "远程信息:"+appConfigName;
    }
    @RequestMapping("/nacosConfig2")
    public String nacosConfig2(){
        return "公共配置:"+appConfigName+",环境配置信息:"+env;
    }
    @RequestMapping("/nacosConfig3")
    public String nacosConfig3(){
        return "全局配置:"+globalConfig+",公共配置:"+appConfigName+",环境配置信息:"+env;
    }
}
