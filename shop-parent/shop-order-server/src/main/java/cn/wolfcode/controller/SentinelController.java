package cn.wolfcode.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.concurrent.TimeUnit;

/**
 * Created by lanxw
 */
@RestController
public class SentinelController {
    @RequestMapping("/sentinel1")
    public String sentinel1(){
        //模拟一次网络延时
        try {
            TimeUnit.SECONDS.sleep(3);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "sentinel1";
    }
    @RequestMapping("/sentinel2")
    public String sentinel2(){
        return "测试高并发下的问题";
    }
    @RequestMapping("/sentinel3")
    public String sentinel3(){
        return "sentinel3";
    }
}
