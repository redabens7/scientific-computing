%% I. Ajustement de la courbe de Keeling au sens des moindres carrés

figure(1)
plot(t,y,'k-');
xlabel('Année');
ylabel('Concentration en CO_2 (ppm)');
title('Courbe de Keeling — CO_2 au Mauna Loa');
grid on  % affiche une grille 

% observation de la courbe : ne semble pas linéaire, plutôt parabolique 
% donc on tente une approximation parabolique 

[Achap, yfit] = approx_parabole(t, y) ; 

a_tilde = Achap(1) ; 
b_tilde = Achap(2) ;
c_tilde = Achap(3) ; 

figure(2)
plot(t, y, 'k-', t, yfit, 'r-') ;
xlabel('Année') ;
ylabel('Concentration en CO_2 (ppm)') ;
title('Courbe de Keeling et ajustement parabolique — CO_2 au Mauna Loa') ;
legend('Données réelles', 'Modèle ajusté') ; 
grid on

% quantification des résidus

residus_absolus = y - yfit;
residus_relatifs = residus_absolus ./ y ; 

figure(3)
plot(t, residus_absolus, 'b-') ;
xlabel('Année') ;
ylabel('Résidus (ppm)') ;
title('Résidus de l’ajustement parabolique') ;
grid on


figure(4)
plot(t, residus_relatifs,'g-') ; 
xlabel('Année') ;
ylabel('Résidus en proportion CO_2') ;
title('Résidus relatifs de l’ajustement parabolique') ;
grid on


% prédiction jusqu'en 2100 

t2100 = 2024:2100 ; 

M2100 = [t2100(:).^2, t2100(:), ones(size(t2100(:)))] ; 

y2100 = M2100 * Achap ;

figure(5)
plot(t, y, 'k-', t, yfit, 'r-', t2100, y2100, 'b--');
xlabel('Année');
ylabel('Concentration en CO_2 (ppm)');
title('Concentration en CO_2, modèle et projection jusqu’en 2100');
legend('Données réelles', 'Modèles parabolique', 'Projection parabolique');
grid on


% en 2100, on prévoit avec ce modèle une concentration en dioxyde de
% carbone de 689.1932 ppm (y2100(end) dans la commande)

% effet d'emballement négligé 

%% II. Prise en compte de la variation annuelle de la concentration en CO2

% variation oscillante => composante parabolique => nouvelle fonction 



[Achap2, yfit2] = approx_parab_sin(t, y) ; 

a2 = Achap2(1);
b2 = Achap2(2);
c2 = Achap2(3);
d = Achap2(4);
e = Achap2(5);

figure(6)
plot(t, y, 'k-', t, yfit2, 'r-');
xlabel('Année');
ylabel('Concentration en CO_2 (ppm)');
title('Courbe de Keeling et ajustement parabolique avec variation saisonnière');
legend('Données réelles', 'Modèle ajusté');
grid on

residus2 = y - yfit2;
figure(7)
plot(t, residus2, 'b-');
xlabel('Année');
ylabel('Résidus (ppm)');
title('Résidus de l’ajustement parabolique avec variation saisonnière');
grid on 


t1 = 0 : 1/365 : 1 ; 

ysin = d*cos(2*pi*t1(:)./T) + e*sin(2*pi*t1(:)./T) ; 

mois = t1 * 12 ; 


figure(8)
plot(mois, ysin, 'm-');
xlabel('Temps (mois)');
ylabel('Variation saisonnière (ppm)');
title('Composante saisonnière du modèle');
grid on


%% Validation hors échantillon 
T = 1 ;
idx  = t < 2016;                                % apprentissage jusqu'en 2015
[A,~] = approx_parab_sin(t(idx), y(idx));
M    = [t(:).^2, t(:), ones(numel(t),1), cos(2*pi*t(:)/T), sin(2*pi*t(:)/T)];
yhat = M*A;
test = t >= 2016;                               % test 2016-fin
rmse_test  = sqrt(mean((y(test) - yhat(test)).^2))
rmse_train = sqrt(mean((y(idx)  - yhat(idx)).^2))