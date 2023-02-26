package cn.wolfcode.service;

import cn.wolfcode.domain.Order;

public interface OrderService {
    Order createOrder(Long productId,Long userId);
}
