package cn.wolfcode.sentinel;

import com.alibaba.csp.sentinel.slots.block.BlockException;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by lanxw
 */
@Slf4j
public class AnnoControllerBlockHandlerClass {
    //注意这里必须使用static修饰方法
    public static String anno1BlockHandler(String name,BlockException ex) {
        log.error("{}", ex);
        return "接口被限流或者降级了==>AnnoControllerBlockHandlerClass";
    }
}
