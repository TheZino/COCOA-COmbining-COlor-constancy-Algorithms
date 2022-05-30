function theta = recovery_error(X,Y)
% X e Y are row vectors
    normX=sqrt(sum(X.^2,2));
    normY=sqrt(sum(Y.^2,2));
    theta=real(acos((sum(X.*Y,2))./(normX.*normY)))/pi*180;