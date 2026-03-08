function [x_est, rmse] = run_kf(F, H, GQG, R, z, x_true)
% Standard Kalman Filter.  GQG=G*Q*G'.
nx=size(F,1); T=size(z,2);
x=zeros(nx,1); P=eye(nx)*10; x_est=zeros(nx,T);
for k=1:T
    xp=F*x; Pp=F*P*F'+GQG;
    K=Pp*H'/(H*Pp*H'+R);
    x=xp+K*(z(:,k)-H*xp);
    P=(eye(nx)-K*H)*Pp; P=(P+P')/2; x_est(:,k)=x;
end
rmse=[];
if ~isempty(x_true)
    err=x_true(:,1:T)-x_est; rmse=sqrt(mean(sum(err.^2,1)));
end
end
