package cn.wolfcode.feign.fallback;

import cn.wolfcode.domain.Product;
import cn.wolfcode.feign.ProductFeignApi;
import org.springframework.stereotype.Component;

/**
 * Created by lanxw
 */
@Component
public class ProductFeignFallBack implements ProductFeignApi {
    @Override
    public Product findByPid(Long pid) {
        Product product = new Product();
        product.setPid(-1L);
        product.setPname("兜底数据");
        product.setPprice(0.0);
        return product;
    }
}
