package cn.wolfcode.service.impl;


import cn.wolfcode.dao.ProductDao;
import cn.wolfcode.domain.Product;
import cn.wolfcode.service.ProductService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ProductServiceImpl implements ProductService {
    @Autowired
    private ProductDao productDao;
    @Override
    public Product findByPid(Long pid) {
        return productDao.findById(pid).get();
    }
}
