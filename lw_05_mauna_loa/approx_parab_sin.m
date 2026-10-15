function [Achap2, yfit2] = approx_parab_sin(t, y) 

   T = 1 ; 

   M = [t(:).^2 , t(:), ones(size(t(:))), cos(2*pi*t(:)./T), sin(2*pi*t(:)./T)] ; 

   Achap2 = M \ y(:) ; % vecteur contenant les coeff optimaux au sens des moindres carrés

   yfit2 = M * Achap2 ; % ordonnées du modèle ajusté 


end 