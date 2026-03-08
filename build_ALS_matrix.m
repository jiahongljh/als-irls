function A_LS = build_ALS_matrix(F, H, G, Q0, R0, N)
% ALS design matrix.  theta=[vech(Q_w); diag(R_v)],  A_LS*theta = b_LS
nx=size(F,1); nz=size(H,1); [~,g]=size(G);
[Pe,~,~]=dare(F',H',G*Q0*G',R0);
K=Pe*H'/(H*Pe*H'+R0);
Fb=F-F*K*H;
if ~all(abs(eig(Fb))<1), warning('Fbar NOT stable'); end
OO=[]; tmp=eye(nx);
for i=1:N, OO=[OO;H*tmp]; tmp=tmp*Fb; end
M1=zeros(nx^2,g^2); idx=1;
for j=1:g, for k=1:g
    II=zeros(g); II(k,j)=1;
    M1(:,idx)=reshape(dlyap(Fb,G*II*G'),[],1); idx=idx+1;
end, end
M2=zeros(nx^2,nz^2); idx=1;
for j=1:nz, for k=1:nz
    II=zeros(nz); II(k,j)=1;
    M2(:,idx)=reshape(dlyap(Fb,F*K*II*K'*F'),[],1); idx=idx+1;
end, end
Gam=eye(nz);
for i=1:N-1, Gam=[Gam;-H*Fb^(i-1)*F*K]; end
OOt=kron(H,OO);
A_LS=[OOt*M1*symtran(g), OOt*M2+kron(eye(nz),Gam)];
A_LS=A_LS(:,[1:g*(g+1)/2, end-nz+1:end]);
if rank(A_LS,1e-6)<size(A_LS,2), warning('A_LS not full column rank!'); end
end
