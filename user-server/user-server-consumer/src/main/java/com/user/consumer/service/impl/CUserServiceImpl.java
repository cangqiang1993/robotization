package com.user.consumer.service.impl;

import com.user.consumer.entity.CUser;
import com.user.consumer.dao.CUserMapper;
import com.user.consumer.service.ICUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

/**
 * <p>
 * 用户表 服务实现类
 * </p>
 *
 * @author cq
 * @since 2023-03-12
 */
@Service
public class CUserServiceImpl extends ServiceImpl<CUserMapper, CUser> implements ICUserService {

}
