# Real-World Multi-Modal and Longitudinal Lung Cancer Dataset

**Abstract.** Multi-modal learning has demonstrated strong potential in
medical applications by integrating heterogeneous data sources such as
medical imaging, clinical records, and genomics to improve predictive
performance and support clinical decision-making. However, advances
in this area are often constrained by two key challenges: the limited
availability of well-curated, ready-to-use datasets that accurately reflect
real-world conditions, where medical data are frequently collected incon-
sistently and are often incomplete; and the inherent difficulty of integrat-
ing heterogeneous data modalities. In this work, we introduce a newly
curated multi-center, multi-modal, and longitudinal dataset designed to
support the evaluation of a wide range of learning pipelines under real-
istic conditions. The dataset comprises a total of 1,365 lung cancer pa-
tients and has three imaging modalities (whole-slide images, CT scans,
and PET scans), structured clinical data, transcriptomic, and longitu-
dinal follow-up and treatment information. For each imaging modality
the dataset contains more than one instance. Moreover, the dataset ex-
hibits substantial and non-uniform missingness across modalities, making
it well-suited for studying robust multi-modal fusion strategies. We fur-
ther provide both uni-modal and multi-modal benchmarks on the task of
12-month overall survival prediction, disease-specific survival, as well as
longitudinal benchmark of hazard prediction under severe missing data.
Our results show that, despite high levels of missingness, integrating
complementary modalities consistently improves predictive performance
over uni-modal approaches, highlighting the value of multi-modal fusion
in realistic clinical settings. 

*This repository contains the benchmark code and the lung cancer features used in our experiments.* 