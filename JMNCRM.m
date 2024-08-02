
load('session1_sub14_eeg.mat')
load('session2_sub14_eeg.mat')
load('session3_sub14_eeg.mat')

load('session1_sub14_label.mat')
load('session2_sub14_label.mat')
load('session3_sub14_label.mat')

for sub=1:1
    Xs_1=session1_sub14_eeg;
    Ys_1=session1_sub14_label;
    
    Xs_2=session2_sub14_eeg;
    Ys_2=session2_sub14_label;
    
    Xt=session3_sub14_eeg;
    Yt=session3_sub14_label;
    
    [~,ns1]=size(Xs_1);
    [~,ns2]=size(Xs_2);
    [~,nt]=size(Xt);

    [Xs_1,~] = mapminmax(Xs_1,0,1);
    [Xs_2,~] = mapminmax(Xs_2,0,1);
    [Xt,~] = mapminmax(Xt,0,1);
    
    Xs_1=Xs_1-repmat(mean(Xs_1,2),[1,ns1]);
    Xs_2=Xs_2-repmat(mean(Xs_2,2),[1,ns2]);
    Xt=Xt-repmat(mean(Xt,2),[1,nt]);
    Xs=[Xs_1 Xs_2];
    X  =[Xs Xt] ; 
    Ys=[Ys_1; Ys_2];
    
    S= Initialization_S(Xs);
    DD=diag(sum(S,2));
    LL=DD-S;       
    XLX=Xs*LL*Xs';       
    
    [~,ns]=size(Xs);
    H=eye(ns)-1/ns * ones(ns,ns);
    F=H*Xs';
    norm_F=1./vecnorm(F);
    A=diag(norm_F)*F'*F*diag(norm_F);
    A=A.^2;          
     
    [d,n]=size(X);
    c=5;
    Y_L_onehot = onehot(Ys,c);
 
    lambdalib = 2.^2;
    betalib = 2.^5 ;
    alphalib = 2.^5;
    best_acc=0;
    best_lambda=-10;
    best_beta=-10;
    best_alpha=-10;
    best_Y=zeros(n,c);
    best_b=zeros(c,1);
    best_W=zeros(d,c);
    
    Y_predict=zeros(nt,1);

    
    maxIter=80;
    % initialize H
    H=centeringMatrix(n);       
    % initialize St    
    St=X*H*X';
    for li=1:length(lambdalib)
        for bi=1:length(betalib)
            for ai=1:length(alphalib)
            lambda=lambdalib(li);
            beta=betalib(bi);
            alpha=alphalib(ai);
            
            % initialize D              
            D=eye(d)/d;
            
            % initialize Y
            Y=ones(n,c)/c;
            Y(1:ns,:)=Y_L_onehot;

            % initialize B   B=2Y-1
            B=2*Y-ones(n,c);
            B(ns+1:end,:)=0;

            % initialize M
            M=zeros(n,c);
                       
            obj=[];  
            acc_iter=[];
            for iter=1:maxIter
                % update W
                J = St+lambda*D+beta*A+alpha*XLX;
                Q = X*H*(Y+B.*M);
                W=J\Q;
                
                % updata D
                D = diag( 0.5./sqrt(sum(W.*W,2)+eps));
                
                % update b
                b=(sum(Y+B.*M,1)'-sum(W'*X,2))/n;

                % update Y
                temp=B.*M;
                for i=ns+1:n
                    Y(i,:)=X(:,i)'*W+b'-temp(i,:);
                    Y(i,:) = EProjSimplex_new(Y(i,:));
                end
                
                % update B
                B=2*Y-ones(n,c);
                %update M
                P=X'*W+ones(n,1)*b'-Y;
                for i=1:n
                    for j=1:c
                        if B(i,j)*P(i,j)>0
                            M(i,j)=P(i,j)/B(i,j);
                        else
                            M(i,j)=0;
                        end
                    end
                end
                
                 % objective function        
                 obj(iter)=F22norm(X'*W+repmat(b',[n 1])-Y-B.*M)+lambda*trace(W'*D*W)+beta*trace(W'*A*W)+alpha*trace(W'*XLX*W);                  
                 
                 R=Y+B.*M;
                 Y_u=R(ns+1:n,:);
                 [~,predict_label] = max(Y_u,[],2);
                 acc = length(find(predict_label == Yt))./nt;                 
                 acc_iter(iter)=acc;

                 if acc>best_acc
                     best_acc=acc;
                     best_lambda=log2(lambda);
                     best_beta=log2(beta);
                     best_alpha=log2(alpha);
                     Y_predict=predict_label;
                     best_b=b;
                     best_B=B;
                     best_M=M;
                     best_W=W;
                     fprintf('sub=%0.4f, best_acc=%0.4f,lambda=%d,beta=%d,alpha=%d\n',sub,best_acc,log2(lambda),log2(beta),log2(alpha));
                 end
            end 
            end
         
        end
    end


end

%%

x=[1:1:80];

figure;
y1=obj;
y2=acc_iter*100;
yyaxis left;
plot(x,y1,'Linewidth',2.2)
%ylabel('Objective function value');
ylabel('Objective value');


yyaxis right
plot(x,y2,'Linewidth',2.2)
%ylabel('Classification accuracy(%)');
ylabel('Acc(%)');
legend('Obj value','Acc (%)')

xlabel('Number of iterations')
axis equal

ax = gca;
set(ax, 'FontSize', 22); 
