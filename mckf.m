function [x_est, P_est, rmse] = mckf(F, H, GQG, R, z, x_true, sigma_k)
% Maximum Correntropy KF (Chen et al., Automatica 76:70-77, 2017).
nx=size(F,1); T=size(z,2);
x=zeros(nx,1); P=eye(nx)*10;
x_est=zeros(nx,T); P_est=zeros(nx,nx,T);
for k=1:T
    xp=F*x; Pp=F*P*F'+GQG;
    x_iter=xp; K=zeros(nx,size(H,1));
    for it=1:10
        e=z(:,k)-H*x_iter;
        w=exp(-norm(e)^2/(2*sigma_k^2)); w=max(w,1e-8);
        K=Pp*H'/(H*Pp*H'+R/w);
        x_new=xp+K*(z(:,k)-H*xp);
        if norm(x_new-x_iter)<1e-5, x_iter=x_new; break; end
        x_iter=x_new;
    end
    x=x_iter; P=(eye(nx)-K*H)*Pp; P=(P+P')/2;
    x_est(:,k)=x; P_est(:,:,k)=P;
end
rmse=[];
if ~isempty(x_true)
    err=x_true(:,1:T)-x_est; rmse=sqrt(mean(sum(err.^2,1)));
end
end
