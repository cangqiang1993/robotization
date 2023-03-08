package com.user.consumer.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RequestMapping("/user")
@RestController
public class UserController {
    @Value("${java.name}")
    private String name;
    @Value("${java.age}")
    private String age;

    @GetMapping("/getConfigTest")
    public String getConfigTest(){
        return "访问成功";
    }
}
