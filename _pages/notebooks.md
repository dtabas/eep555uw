---
layout: archive
title: "Notebooks"
permalink: /notebooks/
author_profile: true
---

{% include base_path %}

{% for post in site.notebooks reversed %}
  {% include archive-single.html %}
{% endfor %}
