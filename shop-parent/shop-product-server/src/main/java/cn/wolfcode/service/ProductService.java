package cn.wolfcode.service;

import cn.wolfcode.domain.Product;

public interface ProductService {
    Product findByPid(Long pid);
}
