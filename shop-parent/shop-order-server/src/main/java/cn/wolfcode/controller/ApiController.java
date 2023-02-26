package cn.wolfcode.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Created by lanxw
 */
@RestController
@RequestMapping("/api1")
public class ApiController {
    @RequestMapping("/hello")
    public String api1(){
        return "api1";
    }
}
