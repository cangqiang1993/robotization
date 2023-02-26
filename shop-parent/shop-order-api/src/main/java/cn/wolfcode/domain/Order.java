package cn.wolfcode.domain;

import lombok.Data;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

//订单
@Entity(name = "t_shop_order")
@Data
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long oid;//订单id

    //用户
    private Long uid;//用户id
    private String username;//用户名
    //商品
    private Long pid;//商品id
    private String pname;//商品名称
    private Double pprice;//商品单价
    //数量
    private Integer number;//购买数量
}