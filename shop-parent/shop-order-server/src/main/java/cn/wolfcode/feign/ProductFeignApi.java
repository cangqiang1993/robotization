package cn.wolfcode.feign;

import cn.wolfcode.domain.Product;
import cn.wolfcode.feign.fallback.ProductFeignFallBack;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * Created by lanxw
 */
//name的名称一定要和订单服务的服务名保持一致
@FeignClient(name = "product-service",fallback = ProductFeignFallBack.class)
public interface ProductFeignApi {
    @RequestMapping("/product/{pid}")
    public Product findByPid(@PathVariable("pid") Long pid);
}
