{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% for item in members %}
{%- if item in ['model_setup', 'error_trap', 'mesh', 'catchments', 'spatialparam', 'spatialstates', 'spatiotemporalstates', 'input_data', 'loi_ouvrage', 'smash_outputs'] %}
	.. autoclass :: {{ module }}.{{ objname }}.{{ item }}

{%- endif -%}
{%- endfor %}

..
   HACK -- meth the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
   .. autosummary::
      :toctree:
      :noindex:
      {% for item in members %}
      {%- if not item.startswith('_') %}
      {%- if not item in ['model_setup', 'error_trap', 'mesh', 'catchments', 'spatialparam', 'spatialstates', 'spatiotemporalstates', 'input_data', 'loi_ouvrage', 'smash_outputs'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
      {%- endif -%}
      {%- endfor %}

	
      
      
