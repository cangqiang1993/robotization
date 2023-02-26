package cn.wolfcode.controller;

import com.alibaba.csp.sentinel.annotation.SentinelResource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Created by lanxw
 */
@RestController
@Slf4j
public class HotSpotController {
    @RequestMapping("/hotSpot1")
    @SentinelResource(value = "hotSpot1")
    public String hotSpot1(Long productId){
        log.info("访问编号为:{}的商品",productId);
        return "hotSpot1";
    }
}
