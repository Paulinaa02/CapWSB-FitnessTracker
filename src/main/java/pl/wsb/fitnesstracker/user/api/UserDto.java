package pl.wsb.fitnesstracker.user.api;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;

/**
 * DTO representing user data exposed by API.
 */
@Getter
@Setter
@AllArgsConstructor
public class UserDto {

    private Long id;
    private String firstName;
    private String lastName;
    private LocalDate birthdate;
    private String email;
}
