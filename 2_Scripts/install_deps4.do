cap ado uninstall ftools
cap ado uninstall reghdfe
cap ado uninstall ivreghdfe
cap ado uninstall require
ssc install require, replace
net install ftools, from("https://raw.githubusercontent.com/sergiocorreia/ftools/master/src/") replace
net install reghdfe, from("https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/src/") replace
net install ivreghdfe, from("https://raw.githubusercontent.com/sergiocorreia/ivreghdfe/master/src/") replace
