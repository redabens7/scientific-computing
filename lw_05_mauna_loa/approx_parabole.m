function [Achap, yfit] = approx_parabole(t, y)

   M = [t(:).^2 , t(:), ones(size(t(:)))] ; % risque : ne fonctionne pas pour un vecteur ligne ou different 
  
                                            % technique pour régler ça : ecrire t(:)

   Achap = M \ y(:) ; % vecteur contenant les coeff optimaux au sens des moindres carrés

   yfit = M * Achap ; % ordonnées du modèle ajusté 


end