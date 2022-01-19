---
title: "Ch01. The Importance of Interconnect Design"
excerpt: "High-Speed Digital System Design"
tagline: "High-Speed Digital System Design"
header:
  overlay_image: /assets/images/unsplash-Umberto.jpg
  overlay_filter: 0.5
  caption: "Photo by [**Umberto**](https://unsplash.com/@umby) on [**Unsplash**](https://unsplash.com/)"
date: 2021-12-02 01:00:00 +0900
categories:
  - High-Speed Digital System Design
tags:
  - High-Speed Digital System Design
---

# 1. The Importance of Interconnct Design

시스템 동작 high frequency화, conductor들이 더 이상 simple wire로 동작하지 않는다. transmission line과 같은 high-frequency effect를 나타낸다. transmission line들이 적절하게 handle되지 않으면 의도치 않게 system timing을 망칠 수 있다. Digital design은 analog world의 복잡도를 가지게 되었다.
modern digital design의 빠른 발전으로 인해 기존에는 필요하지 않았던 지식이 필요하게 되었다. high-speed design은 일종의 신비주의적인 것으로 취급되었다. 문제는 대부분의 references가 너무 추상적이어서 바로 적용하기가 어렵거나 너무 실용적이어서 해당 주제에 대한 이론을 담기가 어려웠다. 이 책에서는 digital design에 초점을 맞추고 필수적인 컨셉을 설명한다. 

## 1.1 THE BASICS
digital design의 basic idea는 1과 0을 나타내는 시그널로 정보 통신을 한다는 것이다. 주로 사다리꼴의 모양의 voltage signal들을 보내거나 받는다. digital signal을 전달하는 conductive path는 `interconnect`라고 부른다. interconnet는 signal을 보내는 chip에서부터 signal을 받는 chip까지의 모든 electrical pathway를 포함한다(chip packages, connectors, sockets 등). intercoonect의 그룹을 `bus`라고 부른다. digital receiver가 high voltage와 low voltage를 구분하는 voltage 영역을 `threshold region`이라고 부른다. 이 영역 안에서는 receiver는 high가 될 수도 있고 low가 될 수도 있다. Silicon에서 실제 switching voltage는 온도, 공급 전압, 공정 산포 등의 변수에 따라 달라질 수 있다. 시스템 디자이너의 관점에서 high-voltage threshold를 `Vih`, low-voltage threshold를 `Vil`라고 부른다. 디자이너는 high voltage 전송 시 어떤 조건에서도 Vih 이하로 전압이 떨어지지 않도록하고, low voltage 전송 시에는 전압이 Vil 아래에서 유지되도록 한다.
digital system의 동작 속도를 극대화하기 위해서는 threshold region에 머무르는 영역을 최소화해야한다. 즉, digital signal의 rise time이나 fall time을 최대한 빠르게 해야한다. Ideal하게는 무한대로 빠른 edge rate가 사용되는 것이 이상적이나, 현실적으로 수백 picoseconds 수준이 한계일 것이다. Fourier analysis를 통해 빠른 edge rate에는 high frequency가 signal spectrum에 포함되는 것을 확인할 수 있을 것이다. 이것이 그 한계에 대한 실마리인데, 모든 conductor는 capacitance, inductance, frequency-dependent resistance를 가지고 있다. high frequency에서는 이 요소들이 무시될 수 없다. 그래서 wire는 더 이상 wire가 아니라 distributed parasitic element가 되며, delay와 transient impedance profile을 가진다. 이는 전달되는 waveform에 distortion과 glitch의 형태로 나타난다.
