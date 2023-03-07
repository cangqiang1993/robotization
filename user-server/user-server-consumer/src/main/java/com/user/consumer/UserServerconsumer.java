package com.user.consumer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@SpringBootApplication
@EnableDiscoveryClient
public class UserServerconsumer {
    public static void main(String[] args) {
        SpringApplication.run(UserServerconsumer.class,args);
    }
}
