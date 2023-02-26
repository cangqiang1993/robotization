package cn.wolfcode.config;

import com.alibaba.csp.sentinel.adapter.spring.webmvc.callback.RequestOriginParser;
import org.springframework.stereotype.Component;

import javax.servlet.http.HttpServletRequest;

/**
 * Created by lanxw
 */
@Component
public class RequestOriginParserDefinition implements RequestOriginParser {
    @Override
    public String parseOrigin(HttpServletRequest request) {
        /**
         *  定义从请求的什么地方获取来源信息
         *  比如我们可以要求所有的客户端需要在请求头中携带来源信息
         */
        String serviceName = request.getParameter("serviceName");
        return serviceName;
    }
}
