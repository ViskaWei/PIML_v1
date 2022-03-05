from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF, TrimmableProcessIF, BoxableProcessIF, ResTunableProcessIF

class BaseSpecProcessIF(ABC):
    """ Base class for process interface for Spec. """
    # @abstractmethod
    # def set_process(self,):
    #     pass

    @abstractmethod
    def set_spec_process_param(self, param):
        pass

    @abstractmethod
    def process_spec(self, spec: StellarSpec):
        pass

    # @abstractmethod

class TrimmableSpecProcessIF(TrimmableProcessIF, BaseSpecProcessIF):
    """ class for trimmable dataIF in wavelength direction. i.e. wave, flux, etc. """
    
    def set_spec_process_param(self, param):
        super().set_process_param(param)

    def process_spec(self, spec: StellarSpec):
        wave = super().process_data(spec.wave)
        spec.set_wave(wave)

        flux = super().process_data(spec.flux)
        spec.set_flux(flux)

        return spec

class BoxableSpecProcessIF(BoxableProcessIF, BaseSpecProcessIF):
    """ class for boxable data i.e flux, parameter, etc. """
    def set_spec_process_param(self, param):
        return super().set_process_param(param)

    def process_spec(self, spec: StellarSpec):
        flux = super().process_data(spec.flux)
        spec.set_flux(flux)

        para = super().process_data(spec.para)
        spec.set_para(para)

        if spec.pdx is not None:
            pdx = super().process_data(spec.pdx)
            spec.set_pdx(pdx)

        return spec

class ResTunableSpecProcessIF(ResTunableProcessIF, BaseSpecProcessIF):
    """ class for res tunable data i.e flux, parameter, etc. """
    def set_spec_process_param(self, param):
        '''
        param: {"step":10}
        '''
        super().set_process_param(param)

    def set_spec_process_model(self, MODEL_TYPES):
        return super().set_process_model(MODEL_TYPES)

    def process_spec(self, spec: StellarSpec):
        wave = super().process_data(spec.wave)
        spec.set_wave(wave)

        flux = super().process_data(spec.flux)
        spec.set_flux(flux)

        return spec